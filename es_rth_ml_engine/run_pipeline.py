import os
import yaml
import pandas as pd

from src.data_io import load_csv_bars, save_parquet, load_parquet
from src.sessions import filter_rth
from src.features import make_features
from src.labels import make_labels
from src.splits import make_walkforward_splits
from src.model import fit_predict_proba
from src.backtest import run_backtest
from src.metrics import summarize_equity

def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs("data", exist_ok=True)

    # 1) Load CSV and write Parquet (one-time, but safe to rerun)
    df = load_csv_bars(
        path=cfg["data"]["input_csv"],
        timestamp_col=cfg["data"]["timestamp_col"],
        tz_input=cfg["data"]["tz_input"],
    )
    save_parquet(df, cfg["data"]["output_parquet"])

    # 2) Load Parquet + filter to RTH (9:30–16:00 ET)
    df = load_parquet(cfg["data"]["output_parquet"])
    df = filter_rth(
        df,
        tz=cfg["session"]["rth_tz"],
        start=cfg["session"]["rth_start"],
        end=cfg["session"]["rth_end"],
    )

    # 3) Features + labels (causal)
    X = make_features(df)
    y = make_labels(df, horizon=cfg["research"]["horizon_bars"])

    # Align
    data = df.join(X, how="inner").join(y, how="inner")
    feature_cols = X.columns.tolist()
    data = data.dropna(subset=feature_cols + ["y_up"])

    # 4) Walk-forward training/testing
    splits = make_walkforward_splits(
        data.index,
        train_days=cfg["research"]["walkforward"]["train_days"],
        test_days=cfg["research"]["walkforward"]["test_days"],
        step_days=cfg["research"]["walkforward"]["step_days"],
        tz=cfg["session"]["rth_tz"],
    )

    # 5) Run folds, collect signals
    all_trades = []
    all_equity = []

    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        train = data.loc[train_idx]
        test = data.loc[test_idx]

        proba = fit_predict_proba(
            X_train=train[feature_cols],
            y_train=train["y_up"],
            X_test=test[feature_cols],
            model_type=cfg["model"]["type"],
        )

        test = test.copy()
        test["p_up"] = proba
        test["signal"] = (test["p_up"] > cfg["model"]["prob_threshold"]).astype(int)  # long-only baseline

        bt = run_backtest(
            bars=test,
            signal_col="signal",
            point_value=cfg["backtest"]["point_value"],
            tick_size=cfg["backtest"]["tick_size"],
            slippage_ticks=cfg["backtest"]["slippage_ticks"],
            commission_per_side=cfg["backtest"]["commission_per_side"],
            max_position=cfg["backtest"]["max_position"],
        )

        bt["fold"] = i
        all_trades.append(bt["trades"])
        all_equity.append(bt["equity_curve"])

        print(f"Fold {i}: trades={len(bt['trades'])}, net_pnl=${bt['equity_curve']['equity'].iloc[-1]:,.2f}")

    trades = pd.concat(all_trades, axis=0).sort_index()
    equity = pd.concat(all_equity, axis=0).sort_index()
    summary = summarize_equity(equity, trades)

    print("\n=== SUMMARY (all folds combined) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Save outputs
    os.makedirs("reports", exist_ok=True)
    trades.to_csv("reports/trades.csv")
    equity.to_csv("reports/equity.csv")
    print("\nWrote reports/trades.csv and reports/equity.csv")

if __name__ == "__main__":
    main()
