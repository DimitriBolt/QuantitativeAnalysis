import numpy as np
import pandas as pd

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"].fillna(0)

    # Log returns
    r1 = np.log(c).diff(1)
    r5 = np.log(c).diff(5)
    r15 = np.log(c).diff(15)

    # Rolling volatility proxy
    vol20 = r1.rolling(20).std()

    # Range features
    rng = (h - l)
    rng20 = rng.rolling(20).mean()
    close_pos = (c - l) / (h - l).replace(0, np.nan)

    # Volume z-score (within rolling window)
    v_z = (v - v.rolling(50).mean()) / v.rolling(50).std()

    # Time-of-day (minute of session) in ET
    et = df.index.tz_convert("America/New_York")
    minutes = et.hour * 60 + et.minute
    # session minute: 9:30 = 570
    session_minute = minutes - (9 * 60 + 30)
    tod_sin = np.sin(2 * np.pi * session_minute / (6.5 * 60))
    tod_cos = np.cos(2 * np.pi * session_minute / (6.5 * 60))

    X = pd.DataFrame(
        {
            "r1": r1,
            "r5": r5,
            "r15": r15,
            "vol20": vol20,
            "rng": rng,
            "rng20": rng20,
            "close_pos": close_pos,
            "v_z": v_z,
            "tod_sin": tod_sin,
            "tod_cos": tod_cos,
        },
        index=df.index,
    )

    # Make sure features are causal: everything here uses current/past only (diff/rolling)
    return X
