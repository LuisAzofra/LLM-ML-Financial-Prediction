"""
price_cache.py — caché on-disk de precios OHLCV por símbolo, causal-safe.

Problema: los backtests masivos re-descargan los mismos símbolos (SPY, QQQ,
TLT, ^IRX, ...) sobre rangos de fechas solapados y golpean los 429 de Yahoo.

Estrategia: cachear la historia COMPLETA canónica de cada símbolo una sola vez
(CACHE_FLOOR → hoy) y servir cada ventana solicitada por slicing en lectura.
Es behavior-preserving porque `download_stock_data` solo descarga+normaliza
OHLCV crudo (sin indicadores): Yahoo devuelve closes independientes de la
ventana y Adj_Close es un alias de Close, así que slicear la historia completa
es idéntico a una descarga fresca de esa ventana.

API pública:
- `cache_path(symbol, interval='1d') -> str`
- `load_cached(symbol, interval='1d') -> pd.DataFrame | None`
- `store_cached(symbol, df_full, interval='1d') -> None`
- `get_or_fetch(symbol, start_date, end_date, interval, fetch_fn) -> pd.DataFrame`
  fetch_fn: callable `(start, end) -> DataFrame` (el body de Yahoo existente).
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import pandas as pd

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'cache',
    'prices',
)
CACHE_FLOOR = '2004-01-01'
STALE_DAYS = 3

_CACHE_ENABLED = os.environ.get('PRICE_CACHE') != '0'


def cache_path(symbol: str, interval: str = '1d') -> str:
    san = symbol.upper().replace('/', '_').replace('^', '_idx_')
    return os.path.join(CACHE_DIR, f"{san}__{interval}.pkl")


def load_cached(symbol: str, interval: str = '1d') -> Optional[pd.DataFrame]:
    path = cache_path(symbol, interval)
    if not os.path.exists(path):
        return None
    return pd.read_pickle(path)


def store_cached(symbol: str, df_full: pd.DataFrame, interval: str = '1d') -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    df_full.to_pickle(cache_path(symbol, interval))


def _slice(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    lo = pd.Timestamp(start_date).normalize()
    hi = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
    mask = (df.index >= lo) & (df.index < hi)
    return df.loc[mask].copy()


def get_or_fetch(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    fetch_fn: Callable[[str, str], pd.DataFrame],
) -> pd.DataFrame:
    if not _CACHE_ENABLED:
        return fetch_fn(start_date, end_date)

    today = pd.Timestamp.now().normalize()
    full_end = (today + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    cached = load_cached(symbol, interval)

    if cached is None or cached.empty:
        full = fetch_fn(CACHE_FLOOR, full_end)
        if full is None or full.empty:
            return full
        store_cached(symbol, full, interval)
        return _slice(full, start_date, end_date)

    req_start = pd.Timestamp(start_date).normalize()
    req_end = pd.Timestamp(end_date).normalize()
    cached_min = cached.index.min().normalize()
    cached_max = cached.index.max().normalize()
    stale_threshold = req_end - pd.tseries.offsets.BDay(STALE_DAYS)

    if cached_max < stale_threshold or req_start < cached_min:
        full = fetch_fn(CACHE_FLOOR, full_end)
        if full is None or full.empty:
            return _slice(cached, start_date, end_date)
        store_cached(symbol, full, interval)
        return _slice(full, start_date, end_date)

    return _slice(cached, start_date, end_date)
