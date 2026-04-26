"""
backfill_news.py — alimenta el cache SQLite (utils/news_cache.py) desde
los feeds RSS actuales (data/news_fetcher.py · Yahoo Finance + cripto).

Limitación honesta: los feeds RSS sólo exponen las noticias más recientes
(~últimos 7-21 días por feed). Para backtests históricos, esto da cobertura
parcial sólo en ventanas que incluyan fechas recientes; el resto cae al
implied sentiment proxy de news_cache.py.

Uso:
    .venv/bin/python tools/backfill_news.py
    .venv/bin/python tools/backfill_news.py AAPL MSFT BTC-USD ETH-USD

Sin args: símbolos por defecto del benchmark.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.news_fetcher import NewsFetcher
from utils.news_cache import NewsCache

DEFAULT_SYMBOLS = [
    # equities populares en el bot
    'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'META', 'SPY', 'QQQ',
    # cripto principales
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD',
]

CRYPTO_SUFFIXES = ('-USD', '-USDT')


def is_crypto(sym: str) -> bool:
    return any(sym.upper().endswith(s) for s in CRYPTO_SUFFIXES)


def backfill(symbols, max_items: int = 30) -> int:
    nf = NewsFetcher()
    cache = NewsCache()
    total = 0
    for sym in symbols:
        try:
            payload = nf.get_cached_news(sym, is_crypto(sym), max_items=max_items)
            items = payload.get('news', []) if isinstance(payload, dict) else []
            inserted = cache.upsert(sym, items)
            count = cache.count_for(sym)
            total += inserted
            print(f"  {sym:<10}  fetched={len(items):3d}  inserted_new={inserted:3d}  total_in_cache={count:4d}")
        except Exception as e:
            print(f"  {sym:<10}  FAIL · {e}")
        # Pequeña pausa para no abusar del feed
        time.sleep(0.3)
    cache.close()
    return total


def main() -> int:
    symbols = sys.argv[1:] or DEFAULT_SYMBOLS
    print(f"Backfilling news cache · {len(symbols)} símbolos · cache={os.path.join('cache','news.db')}")
    t0 = time.time()
    n = backfill(symbols)
    print(f"\nDone. {n} noticias nuevas en {time.time()-t0:.1f}s.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
