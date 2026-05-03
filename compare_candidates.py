"""
Tier 4.2 — A/B harness para validar candidatos C1-C5 antes de mergear.

Reutiliza el patrón de compare_aggressive.py pero parametrizado por
candidato (--cand C1..C5). Cada candidato se prueba ON/OFF sobre las
mismas 12 ventanas seed=42 con baseline AGGR_KELLY_PCT (Tier 4.1).

Aplica gate Agresivo Tier 4.1 (utils/backtest_metrics.aggressive_gate).
Solo si ACEPTA debe mergearse el cambio.

Uso:
    PORT=5057 .venv/bin/python -u api.py &
    .venv/bin/python -u compare_candidates.py --cand C1   # ETFs
    .venv/bin/python -u compare_candidates.py --cand C2   # vol targeting
    .venv/bin/python -u compare_candidates.py --cand C3   # top-N rotation
    .venv/bin/python -u compare_candidates.py --cand C4   # vol filter
    .venv/bin/python -u compare_candidates.py --cand C5   # MR combo
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from urllib import request as urlreq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.backtest_metrics import (
    aggregate_with_ci,
    aggressive_gate,
    format_aggregate_table,
)

API = "http://localhost:5057"
WINDOW_YEARS = 2
SEED = 42

EARLIEST = datetime(2018, 1, 1)
LATEST   = datetime(2024, 4, 1)
DELTA_DAYS = (LATEST - EARLIEST).days

# AGGR_KELLY_PCT (Tier 4.1) — baseline para los candidatos C1-C5
AGGR_BASE = {
    'mode': 'trend',
    'initial_capital': 100_000,
    'allow_short': False,
    'signal_percentile': 0.70,
    'kelly_scale': 0.30,
    'trend_reverse_exit': True,
    'disable_atr_stop': True,
    'max_holding_days': 120,
    'target_vol': 0.20,
    'use_llm': False,
}

# Definición de cada candidato como un override del baseline AGGR.
# El 'on' es el body extra que activa el cambio. 'off' = baseline AGGR.
CANDIDATES = {
    'C1': {
        'name': 'C1 ETFs in watchlist',
        'on':   {'include_etfs': True},
        'off':  {'include_etfs': False},
        'rationale': '5 ETFs (XLE, XLF, GLD, MTUM, IWM) para diversificar tech cluster',
    },
    'C2': {
        'name': 'C2 Volatility-targeting portfolio-level',
        'on':   {'enable_vol_target_overlay': True, 'vol_target_overlay_annual': 0.20},
        'off':  {'enable_vol_target_overlay': False},
        'rationale': 'Moreira-Muir 2017: scale by target_vol/realized_vol → +25% Sharpe',
    },
    'C3': {
        'name': 'C3 Top-N cross-sectional rotation',
        'on':   {'include_etfs': True, 'topn_rotation': True, 'topn_value': 5,
                 'max_concurrent': 5},
        'off':  {'include_etfs': True, 'topn_rotation': False, 'max_concurrent': 3},
        'rationale': 'Han et al cripto + Asness stocks: top-5 por momentum 60d de 15 candidatos',
    },
    'C4': {
        'name': 'C4 Volatility filter (skip low/high vol)',
        'on':   {'vol_filter_min': 0.10, 'vol_filter_max': 1.50},
        'off':  {'vol_filter_min': None,  'vol_filter_max': None},
        'rationale': 'Freqtrade VolatilityFilter: skip si vol_anual fuera [10%, 150%]',
    },
    'C5': {
        'name': 'C5 Mean-reversion combo (ADX/Hurst regime)',
        'on':   {'enable_mr_combo': True, 'mr_adx_max': 18},
        'off':  {'enable_mr_combo': False},
        'rationale': 'RobotWealth + Price Action Lab: ADX<18 → MR (RSI<10), ADX>25 → trend',
    },
}


def random_window(rng):
    days = rng.randint(0, DELTA_DAYS)
    start = EARLIEST + timedelta(days=days)
    end   = start + timedelta(days=365 * WINDOW_YEARS)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def call(start, end, extras):
    body = {**AGGR_BASE, **extras, 'start_date': start, 'end_date': end}
    req = urlreq.Request(
        f"{API}/api/paper/autonomous-backtest",
        method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urlreq.urlopen(req, timeout=900) as r:
        payload = json.loads(r.read())
    elapsed = time.time() - t0
    if payload.get('status') != 'success':
        raise RuntimeError(f"variant {extras} {start}→{end} → {payload.get('error', payload)}")
    p = payload['performance']
    return {
        'return':  p['total_return_pct'],
        'bh':      p['buy_hold_return_pct'],
        'alpha':   p['total_return_pct'] - p['buy_hold_return_pct'],
        'maxdd':   p['max_drawdown_pct'],
        'sharpe':  p['sharpe_ratio'],
        'pf':      p['profit_factor'],
        'wr':      p['win_rate_pct'],
        'trades':  p['total_trades'],
        'elapsed': elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cand', required=True, choices=list(CANDIDATES.keys()))
    parser.add_argument('--n_windows', type=int, default=12)
    args = parser.parse_args()
    cand = CANDIDATES[args.cand]

    rng = random.Random(SEED)
    windows = [random_window(rng) for _ in range(args.n_windows)]
    print(f"Candidato: {cand['name']}")
    print(f"  rationale: {cand['rationale']}")
    print(f"  baseline (off): {cand['off']}")
    print(f"  test     (on):  {cand['on']}")
    print(f"  ventanas (seed={SEED}, N={args.n_windows}):")
    for i, (s, e) in enumerate(windows, 1):
        print(f"    W{i}  {s} → {e}")
    print()

    results_off = []
    results_on  = []
    for i, (s, e) in enumerate(windows, 1):
        print(f"── Ventana {i}/{args.n_windows}  {s}→{e} ──")
        for label, extras, store in [('off', cand['off'], results_off),
                                       ('on',  cand['on'],  results_on)]:
            try:
                m = call(s, e, extras)
                store.append((i, s, e, m))
                print(f"  {args.cand}-{label:3s}  ret={m['return']:+8.2f}%  DD={m['maxdd']:+6.2f}%  PF={m['pf']:5.2f}  trades={m['trades']:3d}  ({m['elapsed']:.1f}s)")
            except Exception as ex:
                print(f"  {args.cand}-{label:3s}  FAIL · {ex}")
        print()

    aggs_off = aggregate_with_ci(results_off, n_resamples=2000, seed=42)
    aggs_on  = aggregate_with_ci(results_on,  n_resamples=2000, seed=42)

    print('═' * 100)
    print(f'AGREGADOS {args.n_windows} VENTANAS')
    print('═' * 100)
    print(format_aggregate_table({f'{args.cand}-off (baseline AGGR)': aggs_off,
                                   f'{args.cand}-on  (test)': aggs_on},
                                  with_ci=True))
    print()

    gate = aggressive_gate(results_off, results_on, n_resamples=4000)
    ds = gate['delta_sharpe'] or {}
    rt = gate['delta_return'] or {}
    dd = gate['delta_maxdd']  or {}

    print('─── Gate Agresivo (cand-on vs cand-off) ───')
    print(f"  Δ_Sharpe = {ds.get('point',0):+.3f}  CI95 [{ds.get('ci_lo',0):+.3f}, {ds.get('ci_hi',0):+.3f}]   "
          f"sharpe_positive={gate['sharpe_positive']}")
    print(f"  Δ_return = {rt.get('point',0):+.2f}pp  CI95 [{rt.get('ci_lo',0):+.2f}, {rt.get('ci_hi',0):+.2f}]   "
          f"return_big_win={gate['return_big_win']}")
    print(f"  Δ_MaxDD  = {dd.get('point',0):+.2f}pp  CI95 [{dd.get('ci_lo',0):+.2f}, {dd.get('ci_hi',0):+.2f}]   "
          f"dd_safe={gate['dd_safe']}  dd_disaster={gate['dd_disaster']}")
    print()
    if gate['accepted']:
        print(f"  ✓✓✓ {args.cand} ACEPTADO. Mergear el cambio. ✓✓✓")
    else:
        print(f"  ✗✗✗ {args.cand} RECHAZADO ({gate['verdict']}). Revertir/no mergear. ✗✗✗")

    out = {
        'candidate': args.cand,
        'name': cand['name'],
        'rationale': cand['rationale'],
        'on_extras': cand['on'],
        'off_extras': cand['off'],
        'n_windows': args.n_windows,
        'windows': windows,
        'aggs_off': aggs_off,
        'aggs_on':  aggs_on,
        'gate':     {k: v for k, v in gate.items() if k not in ('delta_sharpe', 'delta_return', 'delta_maxdd')},
        'delta_sharpe': ds, 'delta_return': rt, 'delta_maxdd': dd,
        'rows_off': [{'window': i, 'start': s, 'end': e, **m} for i, s, e, m in results_off],
        'rows_on':  [{'window': i, 'start': s, 'end': e, **m} for i, s, e, m in results_on],
    }
    out_path = f'/tmp/cand_{args.cand}_result.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  → JSON: {out_path}")


if __name__ == '__main__':
    main()
