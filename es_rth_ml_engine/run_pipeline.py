# run_pipeline.py

from __future__ import annotations

import os
from typing import List, Dict

import yaml
import pandas as pd
from pandas import DataFrame

from src.data_io import (
    YFDownloadSpec,
    download_yf_bars,
    extract_symbol_frame,
    maybe_save_debug_snapshot,
)
from src.sessions import filter_rth
from src.features import make_features
from src.labels import make_labels
from src.splits import make_walkforward_splits
from src.model import fit_predict_proba
from src.backtest import run_backtest
from src.metrics import summarize_equity


def main() -> None:
    """
    In-memory pipeline:
    1) Download 1-minute bars from yfinance
    2) Filter to RTH (9:30–16:00 ET)
    3) Build causal features + labels
    4) Walk-forward splits
    5) Train model + generate signals
    6) Run backtest + summary
    """
    with open("config.yaml", "r") as f:
        cfg: Dict = yaml.safe_load(f)

    # -----------------------------
    # 1) Download bars (in memory)
    # -----------------------------
    yf_spec: YFDownloadSpec = YFDownloadSpec(
        tickers=cfg["data"]["tickers"],
        interval=cfg["data"]["interval"],
        period=cfg["data"]["period"],
        auto_adjust=cfg["data"]["auto_adjust"],
        progress=cfg["data"]["progress"],
    )

    df_raw: DataFrame = download_yf_bars(yf_spec)

    # Optional: save raw snapshot for debugging (OFF by default)
    maybe_save_debug_snapshot(df_raw, cfg["debug"].get("save_raw_snapshot_parquet"))

    # -----------------------------
    # 2) Choose one symbol to research
    # -----------------------------
    symbol: str = cfg["data"]["symbol"]
    df: DataFrame = extract_symbol_frame(df_raw, symbol=symbol)

    # -----------------------------
    # 3) Filter to RTH
    # -----------------------------
    df = filter_rth(
        df,
        tz=cfg["session"]["rth_tz"],
        start=cfg["session"]["rth_start"],
        end=cfg["session"]["rth_end"],
    )

    maybe_save_debug_snapshot(df, cfg["debug"].get("save_rth_snapshot_parquet"))

    # -----------------------------
    # 4) Features + labels (causal)
    # -----------------------------
    X: DataFrame = make_features(df)
    y: DataFrame = make_labels(df, horizon=cfg["research"]["horizon_bars"])

    # Align bars + features + labels
    data: DataFrame = df.join(X, how="inner").join(y, how="inner")
    feature_cols: List[str] = X.columns.tolist()

    # Drop rows with missing values in features/labels
    data = data.dropna(subset=feature_cols + ["y_up"])

    # Optional: save modeling table snapshot
    maybe_save_debug_snapshot(data, cfg["debug"].get("save_model_table_parquet"))

    # --- Diagnostics: how many unique trading days do I actually have? ---
    idx = data.index
    local = idx.tz_convert(cfg["session"]["rth_tz"])
    unique_days = pd.DatetimeIndex(local.normalize().unique()).sort_values()

    print(f"Bars after RTH+features+labels: {len(data):,}")
    print(f"Unique trading days available: {len(unique_days)}")
    if len(unique_days) > 0:
        print(f"Day range: {unique_days[0].date()} -> {unique_days[-1].date()}")
    print(
        "Need at least train_days + test_days = "
        f"{cfg['research']['walkforward']['train_days'] + cfg['research']['walkforward']['test_days']} "
        "unique days."
    )
    # -------------------------------------------

    # -----------------------------
    # 5) Walk-forward splits
    # -----------------------------
    splits = make_walkforward_splits(
        data.index,
        train_days=cfg["research"]["walkforward"]["train_days"],
        test_days=cfg["research"]["walkforward"]["test_days"],
        step_days=cfg["research"]["walkforward"]["step_days"],
        tz=cfg["session"]["rth_tz"],
    )

    if not splits:
        raise ValueError(
            "No walk-forward splits could be created. "
            "Common causes: not enough days in the downloaded sample. "
            "Try increasing period (if allowed) or reducing train/test windows."
        )

    # -----------------------------
    # 6) Run folds, collect outputs
    # -----------------------------
    all_trades: List[DataFrame] = []
    all_equity: List[DataFrame] = []

    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        train: DataFrame = data.loc[train_idx]
        test: DataFrame = data.loc[test_idx]

        proba = fit_predict_proba(
            X_train=train[feature_cols],
            y_train=train["y_up"],
            X_test=test[feature_cols],
            model_type=cfg["model"]["type"],
        )

        test = test.copy()
        test["p_up"] = proba
        test["signal"] = (test["p_up"] > cfg["model"]["prob_threshold"]).astype(int)

        bt = run_backtest(
            bars=test,
            signal_col="signal",
            point_value=float(cfg["backtest"]["point_value"]),
            tick_size=float(cfg["backtest"]["tick_size"]),
            slippage_ticks=int(cfg["backtest"]["slippage_ticks"]),
            commission_per_side=float(cfg["backtest"]["commission_per_side"]),
            max_position=int(cfg["backtest"]["max_position"]),
        )

        bt_trades: DataFrame = bt["trades"].copy()
        bt_equity: DataFrame = bt["equity_curve"].copy()

        bt_trades["fold"] = i
        bt_equity["fold"] = i

        all_trades.append(bt_trades)
        all_equity.append(bt_equity)

        net_pnl_last: float = float(bt_equity["equity"].iloc[-1])
        print(f"Fold {i}: trades={len(bt_trades)}, net_pnl=${net_pnl_last:,.2f}")

    trades: DataFrame = pd.concat(all_trades, axis=0).sort_index()
    equity: DataFrame = pd.concat(all_equity, axis=0).sort_index()

    summary = summarize_equity(equity, trades)

    print("\n=== SUMMARY (all folds combined) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Optional output files (OFF by default)
    out_trades_csv: str | None = cfg["debug"].get("save_trades_csv")
    out_equity_csv: str | None = cfg["debug"].get("save_equity_csv")

    if out_trades_csv:
        os.makedirs(os.path.dirname(out_trades_csv) or ".", exist_ok=True)
        trades.to_csv(out_trades_csv)
        print(f"Wrote {out_trades_csv}")

    if out_equity_csv:
        os.makedirs(os.path.dirname(out_equity_csv) or ".", exist_ok=True)
        equity.to_csv(out_equity_csv)
        print(f"Wrote {out_equity_csv}")


if __name__ == "__main__":
    main()
