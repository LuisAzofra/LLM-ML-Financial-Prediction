"""
Tier 4.1 — Fase 2: validación de la combinación AGGR sobre 12 ventanas.

Tras los sweeps de Fase 1 (tools/sweep_filters_sizing.py), solo
kelly_scale=0.30 superó el gate Agresivo individualmente. Aquí validamos
dos combinaciones sobre 12 ventanas (más datos que las 6 de los sweeps,
mayor power estadístico):

  · MED_NOLLM        — baseline (kelly=0.22, signal_pct=0.82)
  · AGGR_KELLY       — solo kelly_scale=0.30
  · AGGR_KELLY_PCT   — kelly_scale=0.30 + signal_percentile=0.70 (mejor
                       no-ganador en Fase 1, Sharpe ci_lo=+0.01)

Aplica gate Agresivo de Tier 4.1 a cada AGGR vs baseline.

Uso:
    PORT=5057 .venv/bin/python -u api.py &
    .venv/bin/python -u compare_aggressive.py [N_WINDOWS=12]
"""
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
    delta_ci,
    format_aggregate_table,
)

API = "http://localhost:5057"
N_WINDOWS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
WINDOW_YEARS = 2
SEED = 42

EARLIEST = datetime(2018, 1, 1)
LATEST   = datetime(2024, 4, 1)
DELTA_DAYS = (LATEST - EARLIEST).days

MED_BASE = {
    'mode': 'trend',
    'initial_capital': 100_000,
    'allow_short': False,
    'trend_reverse_exit': True,
    'disable_atr_stop': True,
    'max_holding_days': 120,
    'target_vol': 0.20,
    'use_llm': False,
}

VARIANTS = [
    ('MED_NOLLM',      {**MED_BASE, 'signal_percentile': 0.82, 'kelly_scale': 0.22}),
    ('AGGR_KELLY',     {**MED_BASE, 'signal_percentile': 0.82, 'kelly_scale': 0.30}),
    ('AGGR_KELLY_PCT', {**MED_BASE, 'signal_percentile': 0.70, 'kelly_scale': 0.30}),
]


def random_window(rng):
    days = rng.randint(0, DELTA_DAYS)
    start = EARLIEST + timedelta(days=days)
    end   = start + timedelta(days=365 * WINDOW_YEARS)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def call(start, end, extras):
    body = {**extras, 'start_date': start, 'end_date': end}
    req = urlreq.Request(
        f"{API}/api/paper/autonomous-backtest",
        method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urlreq.urlopen(req, timeout=600) as r:
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
    rng = random.Random(SEED)
    windows = [random_window(rng) for _ in range(N_WINDOWS)]
    print(f"Ventanas (seed={SEED}, N={N_WINDOWS}):")
    for i, (s, e) in enumerate(windows, 1):
        print(f"  W{i}  {s} → {e}")
    print()

    results = {name: [] for name, _ in VARIANTS}
    for i, (s, e) in enumerate(windows, 1):
        print(f"── Ventana {i}/{N_WINDOWS}  {s}→{e} ──")
        for name, extras in VARIANTS:
            try:
                m = call(s, e, extras)
                results[name].append((i, s, e, m))
                print(f"  {name:<18}  ret={m['return']:+8.2f}%  DD={m['maxdd']:+6.2f}%  PF={m['pf']:5.2f}  trades={m['trades']:3d}  ({m['elapsed']:.1f}s)")
            except Exception as ex:
                print(f"  {name:<18}  FAIL · {ex}")
        print()

    aggs = {name: aggregate_with_ci(results[name], n_resamples=2000, seed=42)
            for name, _ in VARIANTS}

    print('═' * 100)
    print(f'AGREGADOS {N_WINDOWS} VENTANAS  (CIs 95% bootstrap)')
    print('═' * 100)
    print(format_aggregate_table(aggs, with_ci=True))
    print()

    # Gate Agresivo: AGGR_KELLY vs baseline, AGGR_KELLY_PCT vs baseline
    base_rows = results['MED_NOLLM']
    print('─── Gate Agresivo (cada AGGR vs MED_NOLLM) ───')
    verdicts = {}
    for name, _ in VARIANTS:
        if name == 'MED_NOLLM':
            continue
        rows = results[name]
        gate = aggressive_gate(base_rows, rows, n_resamples=4000)
        ds = gate['delta_sharpe'] or {}
        rt = gate['delta_return'] or {}
        dd = gate['delta_maxdd']  or {}
        print(f"  {name:<18}")
        print(f"    Δ_Sharpe = {ds.get('point',0):+.3f}  CI95 [{ds.get('ci_lo',0):+.3f}, {ds.get('ci_hi',0):+.3f}]   "
              f"sharpe_positive={gate['sharpe_positive']}")
        print(f"    Δ_return = {rt.get('point',0):+.2f}pp  CI95 [{rt.get('ci_lo',0):+.2f}, {rt.get('ci_hi',0):+.2f}]   "
              f"return_big_win={gate['return_big_win']}")
        print(f"    Δ_MaxDD  = {dd.get('point',0):+.2f}pp  CI95 [{dd.get('ci_lo',0):+.2f}, {dd.get('ci_hi',0):+.2f}]   "
              f"dd_safe={gate['dd_safe']}  dd_disaster={gate['dd_disaster']}")
        print(f"    → VEREDICTO: {gate['verdict']}")
        print()
        verdicts[name] = gate

    # Decisión final
    print('═' * 100)
    print('  DECISIÓN FINAL')
    print('═' * 100)
    accepted = [n for n, g in verdicts.items() if g['accepted']]
    if accepted:
        # Si ambos aceptan, preferir el más extenso (KELLY_PCT)
        pref = 'AGGR_KELLY_PCT' if 'AGGR_KELLY_PCT' in accepted else accepted[0]
        print(f"  ✓ {pref} aceptado. Aplicar como variante MED+ por default.")
    else:
        print("  ✗ Ninguna combinación supera el gate Agresivo. Tier 4.1 RECHAZADO honestamente.")
        print("  El cambio kelly_scale=0.30 funcionó en 6 ventanas pero no en 12 — fragility test failed.")

    out = {
        'seed': SEED,
        'n_windows': N_WINDOWS,
        'variants': {name: extras for name, extras in VARIANTS},
        'aggregates': aggs,
        'verdicts': {name: {
            'accepted': v['accepted'], 'verdict': v['verdict'],
            'delta_sharpe_point': (v['delta_sharpe'] or {}).get('point'),
            'delta_return_point': (v['delta_return'] or {}).get('point'),
            'delta_maxdd_point':  (v['delta_maxdd']  or {}).get('point'),
        } for name, v in verdicts.items()},
        'rows': {name: [{'window': i, 'start': s, 'end': e, **m} for i, s, e, m in rs]
                 for name, rs in results.items()},
    }
    with open('/tmp/compare_aggressive_result.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  → JSON: /tmp/compare_aggressive_result.json")


if __name__ == '__main__':
    main()
