"""
Tier 5 — Universo extendido para test de sesgo de supervivencia.

Carga listas reproducibles de tickers (S&P 500 desde el snapshot CSV oficial,
top cryptos por market cap) y expone una función de muestreo aleatorio
determinístico.

Limitación honesta: el snapshot S&P 500 es la composición ACTUAL del índice,
no la histórica. Empresas que quebraron o salieron del índice antes de 2018
no aparecen aquí. Esto introduce un sesgo residual de supervivencia
(componentes débiles del 2018 pueden no estar) que no es resoluble sin
data de constituyentes históricos (CRSP, paid). Documentado en `Tier 5`
de PROGRESS.md.
"""
from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Path al CSV del S&P 500 — descargado una vez desde
# github.com/datasets/s-and-p-500-companies (CC0 license).
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'universe_snapshots')
_SP500_CSV = os.path.join(_DATA_DIR, 'sp500_constituents.csv')

# Top 30 cryptos por capitalización con disponibilidad histórica conocida en
# yfinance desde ~2018-2019. Lista ordenada por relevancia/longevidad de
# data — cuando hagamos sampling con n_crypto < 30, los primeros tienen más
# probabilidad de tener historial completo (4 años de train requeridos).
CRYPTO_TOP30: List[str] = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD',
    'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'DOT-USD', 'LINK-USD',
    'MATIC-USD', 'LTC-USD', 'BCH-USD', 'TRX-USD', 'UNI-USD',
    'ATOM-USD', 'XLM-USD', 'XMR-USD', 'ETC-USD', 'ALGO-USD',
    'FIL-USD', 'NEAR-USD', 'VET-USD', 'AAVE-USD', 'MKR-USD',
    'SAND-USD', 'MANA-USD', 'THETA-USD', 'AXS-USD', 'EGLD-USD',
]


def _load_sp500_tickers() -> List[str]:
    """Lee el CSV de constituyentes S&P 500 y devuelve la lista de símbolos
    en formato yfinance (BRK.B → BRK-B, BF.B → BF-B).
    """
    if not os.path.exists(_SP500_CSV):
        raise FileNotFoundError(
            f"S&P 500 snapshot no encontrado en {_SP500_CSV}. "
            f"Descárgalo con: curl -s 'https://raw.githubusercontent.com/"
            f"datasets/s-and-p-500-companies/main/data/constituents.csv' "
            f"-o {_SP500_CSV}"
        )
    tickers: List[str] = []
    with open(_SP500_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = (row.get('Symbol') or '').strip()
            if not sym:
                continue
            # yfinance usa '-' en lugar de '.' para sub-clases
            sym_yf = sym.replace('.', '-')
            tickers.append(sym_yf)
    return tickers


# Carga perezosa con cache de proceso — ahorra I/O en sweeps repetitivos
_SP500_CACHE: Optional[List[str]] = None


def get_sp500_tickers() -> List[str]:
    global _SP500_CACHE
    if _SP500_CACHE is None:
        _SP500_CACHE = _load_sp500_tickers()
    return list(_SP500_CACHE)


# Famosas — la lista actual del bot (`api.py:2218-2222`). Se mantiene como
# referencia y como `universe_mode='famous'` (default).
FAMOUS_STOCKS: List[str] = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL']
FAMOUS_CRYPTO: List[str] = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD']


@dataclass
class UniverseSample:
    """Resultado de un muestreo. `entries` es la lista pasada al watchlist
    del backtest; `seed`, `n_stocks`, `n_crypto` se persisten en el JSON
    de resultados para reproducibilidad exacta."""
    entries: List[Tuple[str, str]]
    seed: int
    n_stocks: int
    n_crypto: int
    mode: str


def sample_universe(
    mode: str = 'famous',
    seed: int = 42,
    n_stocks: int = 30,
    n_crypto: int = 20,
    exclude_stocks: Optional[List[str]] = None,
) -> UniverseSample:
    """Construye una lista de tuplas (symbol, asset_type) para el watchlist
    del backtest.

    Args:
        mode: 'famous' (6 tech + 4 cripto, default), 'broad_random' (sample
            aleatorio del S&P 500 + top cryptos), o 'broad_curated' (lista
            sectorialmente balanceada — placeholder, ahora delega a famous).
        seed: semilla para `random.sample`. CRÍTICO: persistirla junto al
            resultado, sin esto el experimento no es reproducible.
        n_stocks: tamaño del muestreo de acciones (sólo aplica a broad_random).
        n_crypto: tamaño del muestreo de cryptos (sólo aplica a broad_random;
            con n=20 cogemos top 20 por market cap, no random — son pocas con
            historial suficiente, sample aleatorio inflaría descartes).
        exclude_stocks: tickers a NO incluir nunca (problemáticos en yfinance,
            recién admitidos al índice, etc.).

    Returns:
        `UniverseSample` con los entries listos para `watchlist`.
    """
    if mode == 'famous':
        entries = (
            [(s, 'stock')  for s in FAMOUS_STOCKS] +
            [(c, 'crypto') for c in FAMOUS_CRYPTO]
        )
        return UniverseSample(entries=entries, seed=seed,
                              n_stocks=len(FAMOUS_STOCKS),
                              n_crypto=len(FAMOUS_CRYPTO), mode=mode)

    if mode == 'broad_random':
        rng = random.Random(seed)
        all_stocks = get_sp500_tickers()
        excluded = set((exclude_stocks or []))
        # Excluir las famosas DEL POOL para hacer el test honesto: si las
        # incluyéramos, broad y famous compartirían el solapamiento y se
        # diluiría el efecto del muestreo.
        excluded.update(FAMOUS_STOCKS)
        eligible = [s for s in all_stocks if s not in excluded]
        n = min(n_stocks, len(eligible))
        sampled_stocks = sorted(rng.sample(eligible, n))
        # Cryptos: priorizamos longevidad. n_crypto=20 → top 20 por market
        # cap. Para n_crypto < 30, sample aleatorio sería ruido (las primeras
        # 10-15 de la lista tienen MUCHO más historial que las últimas).
        n_c = min(n_crypto, len(CRYPTO_TOP30))
        if n_c >= 20:
            sampled_crypto = CRYPTO_TOP30[:n_c]
        else:
            # Mezcla: top 10 fijas + sample del resto para no perder longevidad.
            top10 = CRYPTO_TOP30[:10]
            rest_pool = CRYPTO_TOP30[10:]
            extras = rng.sample(rest_pool, max(0, n_c - 10)) if n_c > 10 else []
            sampled_crypto = top10[:min(n_c, 10)] + extras

        entries = (
            [(s, 'stock')  for s in sampled_stocks] +
            [(c, 'crypto') for c in sampled_crypto]
        )
        return UniverseSample(entries=entries, seed=seed,
                              n_stocks=len(sampled_stocks),
                              n_crypto=len(sampled_crypto), mode=mode)

    if mode == 'broad_curated':
        # Reservado para futuro: lista sectorialmente balanceada manual.
        # Por ahora delega a famous para no romper integraciones.
        return sample_universe(mode='famous', seed=seed)

    raise ValueError(f"universe_mode desconocido: {mode!r}. "
                     f"Esperado: 'famous', 'broad_random', 'broad_curated'.")


if __name__ == '__main__':
    # Smoke test — útil para verificar que el CSV está bien y el sampling
    # es determinístico.
    print(f"S&P 500 tickers cargados: {len(get_sp500_tickers())}")
    print(f"  primeros 5: {get_sp500_tickers()[:5]}")
    print(f"  últimos 5:  {get_sp500_tickers()[-5:]}")

    s_famous = sample_universe('famous')
    print(f"\nfamous: {len(s_famous.entries)} entries → {s_famous.entries}")

    s_broad_a = sample_universe('broad_random', seed=42, n_stocks=30, n_crypto=20)
    s_broad_b = sample_universe('broad_random', seed=42, n_stocks=30, n_crypto=20)
    assert s_broad_a.entries == s_broad_b.entries, "seed no determinístico"
    print(f"\nbroad_random seed=42 ({len(s_broad_a.entries)} entries):")
    for sym, atype in s_broad_a.entries:
        print(f"  {atype:>6} {sym}")

    s_broad_c = sample_universe('broad_random', seed=7, n_stocks=30, n_crypto=20)
    assert s_broad_c.entries != s_broad_a.entries, \
        "diferentes seeds deberían dar diferentes muestras"
    print(f"\n✓ smoke OK")
