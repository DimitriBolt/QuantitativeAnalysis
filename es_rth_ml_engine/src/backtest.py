import pandas as pd
import numpy as np

def run_backtest(
    bars: pd.DataFrame,
    signal_col: str,
    point_value: float,
    tick_size: float,
    slippage_ticks: int,
    commission_per_side: float,
    max_position: int,
):
    """
    Long-only baseline:
      signal=1 -> be long 1 contract
      signal=0 -> flat
    Execution rule:
      position at time t applies to return from (t -> t+1) using close-to-close,
      but we assume entry/exit at next bar open approximated via close with slippage.
    For a skeleton, we use close-to-close PnL and charge costs when position changes.
    """
    df = bars.copy()
    df["signal"] = df[signal_col].clip(0, 1) * max_position

    # Position held from next bar (avoid same-bar lookahead)
    df["pos"] = df["signal"].shift(1).fillna(0)

    # Close-to-close return in points
    df["dclose"] = df["close"].diff()

    # Gross PnL ($)
    df["pnl_gross"] = df["pos"] * df["dclose"] * point_value

    # Trades when position changes
    df["pos_change"] = df["pos"].diff().fillna(df["pos"])
    trade_qty = df["pos_change"].abs()

    # Slippage: charge slippage on each position change in ticks
    slip_cost = trade_qty * slippage_ticks * tick_size * point_value

    # Commission: per side (enter or exit). Position change of 1 = one side.
    comm_cost = trade_qty * commission_per_side

    df["costs"] = slip_cost + comm_cost
    df["pnl_net"] = df["pnl_gross"] - df["costs"]

    df["equity"] = df["pnl_net"].cumsum()

    equity_curve = df[["equity", "pnl_net", "pnl_gross", "costs", "pos"]].copy()

    # Trade list (very simple): record timestamps where trades occur
    trades = df.loc[trade_qty > 0, ["pos_change", "close"]].copy()
    trades.rename(columns={"pos_change": "qty_change"}, inplace=True)

    return {
        "equity_curve": equity_curve,
        "trades": trades,
    }
