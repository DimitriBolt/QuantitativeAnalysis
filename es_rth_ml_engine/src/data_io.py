# src/data_io.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, List, Dict

import pandas as pd
from pandas import DataFrame
import yfinance as yf

REQUIRED_OHLCV_COLS: List[str] = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class YFDownloadSpec:
    """
    Specification for downloading intraday bars from Yahoo Finance via yfinance.

    Notes:
    - For interval="1m", Yahoo typically restricts lookback to a short period.
    - Use 'period' for intraday requests to avoid empty returns (e.g. "7d").
    """
    tickers: List[str]
    interval: str = "1m"
    period: str = "7d"  # recommended for 1m
    auto_adjust: bool = False
    progress: bool = False
    group_by: str = "column"  # yfinance default; keeps MultiIndex columns when many tickers


def _normalize_ohlcv_columns(df: DataFrame) -> DataFrame:
    """
    Normalize column names to lowercase: Open->open, High->high, etc.
    """
    col_map: Dict[str, str] = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    out: DataFrame = df.copy()
    out.columns = [col_map.get(str(c), str(c)).lower() for c in out.columns]
    return out


def _ensure_tz_aware_index(df: DataFrame) -> DataFrame:
    """
    Ensure DatetimeIndex is timezone-aware.
    yfinance often returns tz-aware index for intraday, but we guard anyway.
    If timezone-naive, we interpret it as UTC (best-effort default).
    """
    out: DataFrame = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("Expected a pandas DatetimeIndex from yfinance.")
    if out.index.tz is None:
        # Best-effort: interpret as UTC if Yahoo returned naive timestamps.
        out.index = out.index.tz_localize("UTC")
    return out


def download_yf_bars(spec: YFDownloadSpec) -> DataFrame:
    """
    Download bars for one or multiple tickers using yfinance.

    Returns:
        DataFrame with:
        - If one ticker: columns like ['open','high','low','close','volume'] (lowercase)
        - If multiple tickers: MultiIndex columns (PriceField, Ticker) OR (Ticker, PriceField)
          depending on yfinance behavior. We standardize later.

    This function keeps everything in memory (no disk I/O).
    """
    df_raw: DataFrame = yf.download(
        tickers=spec.tickers,
        interval=spec.interval,
        period=spec.period,
        auto_adjust=spec.auto_adjust,
        progress=spec.progress,
        group_by=spec.group_by,
        threads=True,
    )

    if df_raw is None or df_raw.empty:
        raise ValueError(
            "yfinance returned an empty DataFrame. "
            "Common causes: interval/period not allowed, ticker not supported, or rate-limiting."
        )

    df_raw = _ensure_tz_aware_index(df_raw)
    return df_raw


def extract_symbol_frame(df_raw: DataFrame, symbol: str) -> DataFrame:
    """
    Given yfinance output for multiple tickers, extract one symbol into a standard OHLCV frame.

    Handles common yfinance column layouts:
    - MultiIndex columns with levels like ('Open','ESH26.CME') etc.
    - MultiIndex columns with reversed levels ('ESH26.CME','Open').

    Returns:
        DataFrame indexed by timestamp with columns:
        ['open','high','low','close','volume'] (some may be missing if Yahoo doesn't provide).
    """
    if isinstance(df_raw.columns, pd.MultiIndex):
        # Detect which level contains the symbol.
        lvl0 = df_raw.columns.get_level_values(0).astype(str)
        lvl1 = df_raw.columns.get_level_values(1).astype(str)

        if symbol in set(lvl0):
            # Layout: (symbol, field)
            sub: DataFrame = df_raw.loc[:, df_raw.columns.get_level_values(0) == symbol].copy()
            sub.columns = sub.columns.get_level_values(1)
        elif symbol in set(lvl1):
            # Layout: (field, symbol)
            sub = df_raw.loc[:, df_raw.columns.get_level_values(1) == symbol].copy()
            sub.columns = sub.columns.get_level_values(0)
        else:
            raise KeyError(f"Symbol '{symbol}' not found in yfinance MultiIndex columns.")
    else:
        # Single ticker case: df_raw already looks like OHLCV
        sub = df_raw.copy()

    sub = _normalize_ohlcv_columns(sub)
    sub = _ensure_tz_aware_index(sub)

    # Keep only standard OHLCV columns if present
    keep: List[str] = [c for c in REQUIRED_OHLCV_COLS if c in sub.columns]
    if not keep:
        raise ValueError(f"No OHLCV columns found for '{symbol}'. Columns={list(sub.columns)}")

    out: DataFrame = sub[keep].copy()

    # Basic numeric coercion
    for c in keep:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_index().dropna(subset=["open", "high", "low", "close"])
    return out


def maybe_save_debug_snapshot(df: DataFrame, path: Optional[str]) -> None:
    """
    Optional debug persistence. Use ONLY if path is provided.
    This is not used by default to keep everything in memory.
    """
    if path is None:
        return
    # Use Parquet if you want; CSV also works for quick inspection.
    # Here we choose Parquet (fast + preserves dtypes).
    df.to_parquet(path)
