"""
news_cache.py — caché SQLite de noticias + implied sentiment proxy.

Tier 1.2: reemplaza el hardcode `sent = "Neutral (sin feed de noticias en backtest)"`
en `_recommend_llm_gate` (api.py:2221) por información sustantiva.

Estrategia:
- **Cache real**: SQLite persistente alimentado desde `data.news_fetcher.NewsFetcher`
  (Yahoo Finance RSS + CoinDesk/Cointelegraph). Útil para backtests recientes
  (cobertura efectiva ~últimos 14 días por feeds RSS) y live trading.
- **Implied sentiment proxy**: cuando NO hay noticias en el rango temporal
  relevante (caso típico en backtests históricos 2018-2023), construye una
  descripción textual del estado de mercado a partir de retornos recientes,
  RSS, SMA cross, ATR/vol. Estrictamente mejor que "Neutral" porque captura
  el sentimiento implícito que el precio ya refleja causalmente.

API pública:
- `NewsCache(db_path)`: gestiona SQLite
  - `init_db()`, `upsert(symbol, items)`, `count_for(symbol)`
  - `get_news_before(symbol, before_date, days_window=21, limit=5)`
- `implied_sentiment_from_market(...)` → (label, summary_str)
- `build_sentiment_string(symbol, current_date, day_dict, df_history, news_cache)`
  → string listo para `LLMClient.interpret_market_data`
"""
from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'cache',
    'news.db',
)


@dataclass(frozen=True)
class NewsItem:
    symbol: str
    published_at: str  # ISO 8601
    title: str
    summary: str
    source: str
    link: str

    @classmethod
    def from_fetcher_dict(cls, symbol: str, d: Dict[str, Any]) -> 'NewsItem':
        return cls(
            symbol=symbol.upper(),
            published_at=str(d.get('published', '') or '')[:25],
            title=str(d.get('title', '') or '')[:500],
            summary=str(d.get('summary', '') or '')[:1000],
            source=str(d.get('source', '') or 'unknown')[:80],
            link=str(d.get('link', '') or '')[:500],
        )


class NewsCache:
    """Caché SQLite de noticias por símbolo + fecha de publicación."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS news (
        symbol        TEXT NOT NULL,
        published_at  TEXT NOT NULL,
        title         TEXT NOT NULL,
        summary       TEXT,
        source        TEXT,
        link          TEXT,
        hash          TEXT PRIMARY KEY
    );
    CREATE INDEX IF NOT EXISTS idx_news_symbol_date ON news(symbol, published_at);
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        # Lock que serializa el acceso a la conexión única compartida; api.py es
        # un servidor Flask multihilo y sqlite3 no permite usar una conexión
        # desde varios hilos sin esto.
        self._lock = threading.Lock()
        self.init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            # check_same_thread=False: la conexión se comparte entre hilos de
            # Flask; el acceso queda serializado por self._lock.
            self._conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            self._conn.execute('PRAGMA journal_mode=WAL;')
            self._conn.execute('PRAGMA synchronous=NORMAL;')
        return self._conn

    def init_db(self) -> None:
        with self._lock:
            c = self._get_conn()
            c.executescript(self.SCHEMA)
            c.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @staticmethod
    def _hash(symbol: str, published_at: str, title: str) -> str:
        import hashlib
        h = hashlib.sha1(f"{symbol}|{published_at}|{title}".encode('utf-8'))
        return h.hexdigest()

    def upsert(self, symbol: str, items: Sequence[Dict[str, Any]]) -> int:
        """Inserta o ignora (por hash) una lista de noticias. Devuelve nº insertadas."""
        if not items:
            return 0
        rows = []
        for d in items:
            ni = NewsItem.from_fetcher_dict(symbol, d)
            if not ni.title or not ni.published_at:
                continue
            rows.append((
                ni.symbol, ni.published_at, ni.title, ni.summary, ni.source, ni.link,
                self._hash(ni.symbol, ni.published_at, ni.title),
            ))
        if not rows:
            return 0
        with self._lock:
            c = self._get_conn()
            cur = c.executemany(
                'INSERT OR IGNORE INTO news VALUES (?, ?, ?, ?, ?, ?, ?)', rows,
            )
            c.commit()
            return cur.rowcount or 0

    def count_for(self, symbol: str) -> int:
        with self._lock:
            c = self._get_conn()
            cur = c.execute('SELECT COUNT(*) FROM news WHERE symbol = ?', (symbol.upper(),))
            return int(cur.fetchone()[0])

    def get_news_before(
        self,
        symbol: str,
        before_date: str,
        days_window: int = 21,
        limit: int = 5,
    ) -> List[Dict[str, str]]:
        """
        Devuelve hasta `limit` noticias de `symbol` publicadas en
        [before_date − days_window, before_date), ordenadas DESC por fecha.

        before_date: 'YYYY-MM-DD' (inclusive como cota superior estricta).
        """
        try:
            anchor = datetime.fromisoformat(before_date[:10])
        except ValueError:
            return []
        lower = (anchor - timedelta(days=days_window)).strftime('%Y-%m-%d')
        upper = anchor.strftime('%Y-%m-%d')
        with self._lock:
            c = self._get_conn()
            cur = c.execute(
                """SELECT symbol, published_at, title, summary, source, link
                   FROM news
                   WHERE symbol = ?
                     AND substr(published_at, 1, 10) >= ?
                     AND substr(published_at, 1, 10) <  ?
                   ORDER BY published_at DESC
                   LIMIT ?""",
                (symbol.upper(), lower, upper, limit),
            )
            rows = cur.fetchall()
        return [
            {'symbol': r[0], 'published_at': r[1], 'title': r[2],
             'summary': r[3] or '', 'source': r[4] or '', 'link': r[5] or ''}
            for r in rows
        ]


def _classify(label_score: float) -> str:
    if label_score >= 0.7:
        return 'STRONG_BULLISH'
    if label_score >= 0.3:
        return 'BULLISH'
    if label_score <= -0.7:
        return 'STRONG_BEARISH'
    if label_score <= -0.3:
        return 'BEARISH'
    return 'NEUTRAL'


def implied_sentiment_from_market(
    ret_5d_pct: float,
    ret_20d_pct: float,
    rsi: Optional[float],
    sma50: Optional[float],
    sma200: Optional[float],
    vol_pct: float,
    atr_pct_of_close: Optional[float] = None,
) -> Tuple[str, str]:
    """
    Construye un descriptor textual del sentimiento implícito por el mercado.

    Hipótesis: el precio ya refleja causalmente las noticias relevantes (forma
    eficiente débil), así que retornos recientes + indicadores técnicos sirven
    como proxy honesto cuando no tenemos noticias históricas.

    Devuelve (label, summary_str). Label ∈ STRONG_BEARISH..STRONG_BULLISH.
    """
    score = 0.0
    parts: List[str] = []

    # Retornos recientes (peso alto)
    if ret_5d_pct >= 8:
        score += 0.5
        parts.append(f"strong 5d momentum +{ret_5d_pct:.1f}%")
    elif ret_5d_pct >= 3:
        score += 0.25
        parts.append(f"positive 5d momentum +{ret_5d_pct:.1f}%")
    elif ret_5d_pct <= -8:
        score -= 0.5
        parts.append(f"sharp 5d selloff {ret_5d_pct:.1f}%")
    elif ret_5d_pct <= -3:
        score -= 0.25
        parts.append(f"negative 5d momentum {ret_5d_pct:.1f}%")
    else:
        parts.append(f"flat 5d {ret_5d_pct:+.1f}%")

    if ret_20d_pct >= 12:
        score += 0.25
        parts.append(f"strong 20d trend +{ret_20d_pct:.1f}%")
    elif ret_20d_pct <= -12:
        score -= 0.25
        parts.append(f"weak 20d trend {ret_20d_pct:.1f}%")

    # RSI
    if rsi is not None:
        if rsi >= 70:
            score -= 0.10  # overbought es ligeramente bajista
            parts.append(f"RSI={rsi:.0f} overbought")
        elif rsi <= 30:
            score += 0.10  # oversold puede ser bullish (mean reversion)
            parts.append(f"RSI={rsi:.0f} oversold")
        else:
            parts.append(f"RSI={rsi:.0f}")

    # SMA cross (proxy de tendencia estructural)
    if sma50 is not None and sma200 is not None and sma200 > 0:
        ratio = sma50 / sma200 - 1.0
        if ratio > 0.02:
            score += 0.20
            parts.append(f"SMA50/200 uptrend (+{ratio*100:.1f}%)")
        elif ratio < -0.02:
            score -= 0.20
            parts.append(f"SMA50/200 downtrend ({ratio*100:.1f}%)")

    # Volatilidad
    if vol_pct >= 60:
        parts.append(f"very high vol {vol_pct:.0f}%")
    elif vol_pct >= 35:
        parts.append(f"elevated vol {vol_pct:.0f}%")
    if atr_pct_of_close is not None and atr_pct_of_close >= 5:
        parts.append(f"ATR/price={atr_pct_of_close:.1f}%")

    label = _classify(score)
    summary = (
        f"Implied market sentiment: {label} (score={score:+.2f}). "
        + "; ".join(parts) + "."
    )
    return label, summary


def format_news_for_prompt(items: Sequence[Dict[str, Any]]) -> str:
    """Convierte una lista de noticias en bloque conciso para el LLM (≤ ~600 chars)."""
    if not items:
        return ''
    lines = []
    for it in items[:5]:
        date = str(it.get('published_at', ''))[:10]
        title = str(it.get('title', ''))[:140]
        lines.append(f"  [{date}] {title}")
    return "Recent headlines (within ~21 days before decision):\n" + "\n".join(lines)


def build_sentiment_string(
    symbol: str,
    current_date: str,
    ret_5d_pct: float,
    ret_20d_pct: float,
    rsi: Optional[float],
    sma50: Optional[float],
    sma200: Optional[float],
    vol_pct: float,
    atr_pct_of_close: Optional[float] = None,
    news_cache: Optional[NewsCache] = None,
) -> Tuple[str, str, int]:
    """
    Construye el string de sentimiento listo para `interpret_market_data`.

    Estrategia: SIEMPRE incluye implied sentiment (proxy desde mercado), y si
    encontramos noticias reales en cache las añadimos como contexto adicional.
    Devuelve (sentiment_string, label, n_real_news_used).
    """
    label, implied = implied_sentiment_from_market(
        ret_5d_pct, ret_20d_pct, rsi, sma50, sma200, vol_pct, atr_pct_of_close,
    )

    real_news_block = ''
    n_real = 0
    if news_cache is not None:
        try:
            items = news_cache.get_news_before(symbol, current_date, days_window=21, limit=5)
            n_real = len(items)
            if items:
                real_news_block = "\n" + format_news_for_prompt(items)
        except Exception:
            pass

    return implied + real_news_block, label, n_real
