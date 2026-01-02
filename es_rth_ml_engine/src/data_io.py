import pandas as pd

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

def load_csv_bars(path: str, timestamp_col: str, tz_input: str) -> pd.DataFrame:
    """
    Expects CSV columns: timestamp, open, high, low, close, volume
    timestamp can be timezone-naive or tz-aware; tz_input tells us how to interpret.
    """
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if timestamp_col not in df.columns:
        raise ValueError(f"CSV missing timestamp column '{timestamp_col}'")

    ts = pd.to_datetime(df[timestamp_col], utc=False, errors="coerce")
    if ts.isna().any():
        bad = df.loc[ts.isna(), timestamp_col].head(5).tolist()
        raise ValueError(f"Could not parse some timestamps. Examples: {bad}")

    # Localize or convert
    if ts.dt.tz is None:
        # timezone-naive; interpret as tz_input
        ts = ts.dt.tz_localize(tz_input)
    else:
        # tz-aware; convert to tz_input if user wants consistent base
        ts = ts.dt.tz_convert(tz_input)

    df = df.drop(columns=[timestamp_col]).copy()
    df.index = ts
    df.index.name = "timestamp"

    # Basic numeric clean
    for c in REQUIRED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df

def save_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, engine="pyarrow")

def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if df.index.tz is None:
        raise ValueError("Parquet index must be tz-aware. Rebuild parquet from CSV with tz_input set.")
    return df.sort_index()
