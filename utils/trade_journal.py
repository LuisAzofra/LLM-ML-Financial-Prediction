"""
trade_journal.py — Tier 3.2: memoria persistente de trades + k-NN retrieval.

Inspirado en HKUDS/Vibe-Trading FTS5 memory cross-session. Persiste cada
trade cerrado con un vector de features + metadata para que decisiones
futuras puedan retrievar trades históricos similares y aprender de
ellos (mitigando sesgos cognitivos como momentum chasing y anchoring).

API pública:
- `TradeJournal(db_path)`: gestiona SQLite con tabla `trades`
  - `log_trade(symbol, entry_date, exit_date, action, features_vec,
               pnl_pct, hold_days, regime='', rationale='')`
  - `find_similar(features_vec, k=3, symbol=None, min_examples=10)`
    → list de dicts con metadata + cosine_sim, ordered desc
  - `count(symbol=None)`, `recent_pnl_stats(symbol, n=20)`
- `format_lessons_for_prompt(items, max_chars=600)`: string conciso
  para inyectar al prompt del LLM (Tier 2.3 debate o single-pass)

Diseño anti-leakage: los features se almacenan al ENTRY (no al exit),
así un retrieval futuro no incluye información que no estaba disponible
al momento de la decisión.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'cache',
    'trade_journal.db',
)


class TradeJournal:
    """SQLite-backed memoria de trades cerrados con retrieval k-NN."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS trades (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol        TEXT NOT NULL,
        entry_date    TEXT NOT NULL,
        exit_date     TEXT,
        action        TEXT NOT NULL,
        features_json TEXT NOT NULL,  -- vector como JSON list[float]
        feature_dim   INTEGER NOT NULL,
        pnl_pct       REAL,
        hold_days     INTEGER,
        regime        TEXT,
        rationale     TEXT,
        created_at    TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_trades_symbol_date ON trades(symbol, entry_date);
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self.init_db()

    def _conn_get(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            self._conn.execute('PRAGMA journal_mode=WAL;')
            self._conn.execute('PRAGMA synchronous=NORMAL;')
        return self._conn

    def init_db(self) -> None:
        with self._lock:
            c = self._conn_get()
            c.executescript(self.SCHEMA)
            c.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ── Logging ───────────────────────────────────────────────────────────
    def log_trade(
        self,
        symbol: str,
        entry_date: str,
        exit_date: str,
        action: str,                        # LONG / SHORT
        features_vec: Sequence[float],      # vector compactado al entry
        pnl_pct: float,
        hold_days: int,
        regime: str = '',
        rationale: str = '',
    ) -> int:
        """Inserta un trade cerrado. Devuelve id de la fila."""
        vec = [float(x) if x is not None and not (isinstance(x, float) and math.isnan(x)) else 0.0
               for x in features_vec]
        with self._lock:
            c = self._conn_get()
            cur = c.execute(
                """INSERT INTO trades
                   (symbol, entry_date, exit_date, action, features_json,
                    feature_dim, pnl_pct, hold_days, regime, rationale)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol.upper(), entry_date, exit_date, action.upper(),
                 json.dumps(vec), len(vec), float(pnl_pct), int(hold_days),
                 regime[:64], rationale[:500]),
            )
            c.commit()
            return int(cur.lastrowid)

    def count(self, symbol: Optional[str] = None) -> int:
        with self._lock:
            c = self._conn_get()
            if symbol:
                cur = c.execute('SELECT COUNT(*) FROM trades WHERE symbol = ?', (symbol.upper(),))
            else:
                cur = c.execute('SELECT COUNT(*) FROM trades')
            return int(cur.fetchone()[0])

    def recent_pnl_stats(self, symbol: Optional[str] = None, n: int = 20) -> Dict[str, float]:
        """Stats de los últimos N trades (cerrados): avg_pnl, win_rate, n."""
        with self._lock:
            c = self._conn_get()
            if symbol:
                cur = c.execute(
                    'SELECT pnl_pct FROM trades WHERE symbol = ? ORDER BY id DESC LIMIT ?',
                    (symbol.upper(), n),
                )
            else:
                cur = c.execute('SELECT pnl_pct FROM trades ORDER BY id DESC LIMIT ?', (n,))
            rows = [r[0] for r in cur.fetchall()]
        if not rows:
            return {'n': 0, 'avg_pnl': 0.0, 'win_rate': 0.0}
        wins = sum(1 for p in rows if p > 0)
        return {
            'n': len(rows),
            'avg_pnl':  float(sum(rows) / len(rows)),
            'win_rate': float(wins / len(rows)),
        }

    # ── k-NN retrieval ───────────────────────────────────────────────────
    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b):
            return 0.0
        num = sum(x * y for x, y in zip(a, b))
        da  = math.sqrt(sum(x * x for x in a))
        db  = math.sqrt(sum(y * y for y in b))
        if da < 1e-12 or db < 1e-12:
            return 0.0
        return float(num / (da * db))

    def find_similar(
        self,
        features_vec: Sequence[float],
        k: int = 3,
        symbol: Optional[str] = None,
        before_date: Optional[str] = None,    # anti-leak: sólo trades anteriores
        min_examples: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        k-NN cosine similarity sobre el feature vector. Si `symbol`,
        prioriza misma symbol. Si total < min_examples → devuelve [].
        Si `before_date`, filtra `entry_date < before_date` (anti-leak en backtest).
        """
        if features_vec is None or len(features_vec) == 0:
            return []
        params: List[Any] = []
        where = []
        if symbol:
            where.append('symbol = ?'); params.append(symbol.upper())
        if before_date:
            where.append('entry_date < ?'); params.append(before_date)
        where.append('feature_dim = ?'); params.append(len(features_vec))
        sql = ('SELECT id, symbol, entry_date, exit_date, action, features_json, '
               'pnl_pct, hold_days, regime, rationale FROM trades')
        if where:
            sql += ' WHERE ' + ' AND '.join(where)
        sql += ' ORDER BY id DESC LIMIT 500'
        with self._lock:
            c = self._conn_get()
            cur = c.execute(sql, params)
            rows = cur.fetchall()
        if len(rows) < min_examples:
            return []

        target = list(features_vec)
        scored: List[Dict[str, Any]] = []
        for r in rows:
            try:
                vec = json.loads(r[5])
            except Exception:
                continue
            sim = self._cosine(target, vec)
            scored.append({
                'id':         r[0], 'symbol': r[1], 'entry_date': r[2], 'exit_date': r[3],
                'action':     r[4], 'pnl_pct': r[6], 'hold_days':  r[7],
                'regime':     r[8] or '', 'rationale': r[9] or '',
                'cosine_sim': sim,
            })
        scored.sort(key=lambda x: x['cosine_sim'], reverse=True)
        return scored[:k]


def format_lessons_for_prompt(
    items: Sequence[Dict[str, Any]],
    max_chars: int = 600,
) -> str:
    """
    Renderiza top-k trades similares como bloque conciso para inyectar
    al prompt del LLM. Ejemplo de output:

        Past similar trades (cosine sim, P&L outcome):
          [BTC-USD 2023-04-15 → 2023-04-22] LONG sim=0.91 pnl=+5.4% (regime: bull)
          [BTC-USD 2022-11-08 → 2022-11-15] LONG sim=0.87 pnl=-2.1% (regime: bear_recovery)
          [BTC-USD 2023-08-01 → 2023-08-09] LONG sim=0.83 pnl=+1.8% (regime: chop)
    """
    if not items:
        return ''
    lines = ['Past similar trades (cosine sim, P&L outcome):']
    for it in items[:5]:
        sym  = it.get('symbol', '?')
        ed   = it.get('entry_date', '?')
        xd   = it.get('exit_date', '')
        act  = it.get('action', '?')
        sim  = it.get('cosine_sim', 0.0)
        pnl  = it.get('pnl_pct', 0.0)
        reg  = (it.get('regime', '') or '').lower()[:30]
        line = (f"  [{sym} {ed} → {xd}] {act} sim={sim:.2f} pnl={pnl:+.1f}%"
                + (f" (regime: {reg})" if reg else ''))
        lines.append(line)
    txt = '\n'.join(lines)
    if len(txt) > max_chars:
        txt = txt[:max_chars - 3] + '...'
    return txt
