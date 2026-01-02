import pandas as pd
import numpy as np

def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min())

def summarize_equity(equity_df: pd.DataFrame, trades: pd.DataFrame) -> dict:
    eq = equity_df["equity"]
    rets = equity_df["pnl_net"]

    # “Per-bar Sharpe” scaled to per-day-ish is not ideal; this is just a quick sanity metric.
    # For RTH 1-min bars, ~390 minutes/day.
    if rets.std() > 0:
        sharpe = (rets.mean() / rets.std()) * np.sqrt(390)
    else:
        sharpe = 0.0

    return {
        "Net PnL ($)": f"{eq.iloc[-1]:,.2f}",
        "Max Drawdown ($)": f"{_max_drawdown(eq):,.2f}",
        "Trades (position changes)": str(len(trades)),
        "Sharpe (rough, per-day scaled)": f"{sharpe:.2f}",
    }
