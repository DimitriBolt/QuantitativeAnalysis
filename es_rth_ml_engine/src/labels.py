import pandas as pd

def make_labels(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    y_up = 1 if close[t+h] > close[t], else 0
    """
    future = df["close"].shift(-horizon)
    y_up = (future > df["close"]).astype(int)
    return pd.DataFrame({"y_up": y_up}, index=df.index)
