import pandas as pd

def _unique_trading_days(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    local = index.tz_convert(tz)
    days = pd.DatetimeIndex(local.normalize().unique()).sort_values()
    return days

def make_walkforward_splits(index: pd.DatetimeIndex, train_days: int, test_days: int, step_days: int, tz: str):
    """
    Returns list of (train_index, test_index) timestamp indices, split by unique trading days.
    """
    days = _unique_trading_days(index, tz=tz)
    splits = []

    start = 0
    while True:
        train_start = start
        train_end = train_start + train_days
        test_end = train_end + test_days

        if test_end > len(days):
            break

        train_days_slice = days[train_start:train_end]
        test_days_slice = days[train_end:test_end]

        # Map days back to timestamps
        local = index.tz_convert(tz)
        train_mask = local.normalize().isin(train_days_slice)
        test_mask = local.normalize().isin(test_days_slice)

        train_idx = index[train_mask]
        test_idx = index[test_mask]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

        start += step_days

    return splits
