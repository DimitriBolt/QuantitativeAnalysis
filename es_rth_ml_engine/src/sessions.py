import pandas as pd

def filter_rth(df: pd.DataFrame, tz: str, start: str, end: str) -> pd.DataFrame:
    """
    Keep only 9:30–16:00 ET bars (inclusive start, exclusive end).
    Assumes df.index is tz-aware.
    """
    if df.index.tz is None:
        raise ValueError("df.index must be tz-aware")

    local = df.copy()
    local.index = local.index.tz_convert(tz)

    # Use between_time for intraday window
    local = local.between_time(start, end, inclusive="left")

    # Convert back to original tz of df
    local.index = local.index.tz_convert(df.index.tz)
    return local
