"""
Tier 5 — Validación de la variante SWING vs AGGR_PLUS.

Mismo patrón que compare_aggressive.py: N ventanas aleatorias seed=42, gate
Agresivo (Δ_Sharpe IC95% > 0 Y (Δ_ret IC95% > +5pp O Δ_DD ≤ +2pp)). Por
default ventanas de 6 meses (SWING_YEARS=0.5) — el horizonte donde swing
puede aportar más sobre AGGR_PLUS (que brilla en 1-2y, sufre en 3-6M).

Variantes comparadas:
  · AGGR_PLUS_FAMOUS — baseline Tier 4.2 sobre 6 famosas + 4 cryptos + ETFs
  · SWING_FAMOUS    — mismos defaults + mode='swing' + holding 1-14d
  · SWING_BROAD_42  — SWING + universo broad_random seed=42 (test combinado
                       de Tier 5: sesgo de supervivencia + plazo corto)

Uso:
    PORT=5057 .venv/bin/python -u api.py &
    .venv/bin/python -u compare_swing.py [N_WINDOWS=24]
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
    format_aggregate_table,
)

API = "http://localhost:5057"
N_WINDOWS = int(sys.argv[1]) if len(sys.argv) > 1 else 24
WINDOW_YEARS = float(os.environ.get('SWING_WIN_YEARS', '0.5'))
SEED = int(os.environ.get('SWING_SEED', '42'))

EARLIEST = datetime(2018, 1, 1)
LATEST   = datetime(2024, 4, 1)
DELTA_DAYS = (LATEST - EARLIEST).days

# AGGR_PLUS — Tier 4.2 baseline (validado, ACEPTADO sobre 24 ventanas 2y)
AGGR_PLUS = {
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
    'topn_rotation': True,
    'topn_value': 5,
    'max_concurrent': 5,
    'include_etfs': True,
}

# SWING — Tier 5: mismas que AGGR_PLUS pero con mode='swing' y holding corto.
# atr_stop ON automáticamente en swing (defensa de quiebra).
SWING = {**AGGR_PLUS,
         'mode': 'swing',
         'max_holding_days_swing': 14,
         'min_holding_days_swing': 1,
         'conf_floor': 0.50}

# SWING + universo amplio (test combinado: validar swing en universo no-sesgado)
SWING_BROAD = {**SWING,
               'universe_mode': 'broad_random',
               'universe_seed': 42,
               'universe_size_stocks': 30,
               'universe_size_crypto': 20}

VARIANTS = [
    ('AGGR_PLUS_FAMOUS', AGGR_PLUS),
    ('SWING_FAMOUS',     SWING),
    ('SWING_BROAD_42',   SWING_BROAD),
]


def random_window(rng):
    days = rng.randint(0, DELTA_DAYS - int(365 * WINDOW_YEARS))
    start = EARLIEST + timedelta(days=days)
    end   = start + timedelta(days=int(365 * WINDOW_YEARS))
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
        raise RuntimeError(f"variant {extras.get('mode','?')} {start}→{end} → "
                           f"{payload.get('error', payload)}")
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
        'avg_hold': p.get('avg_hold_days', 0),
        'elapsed': elapsed,
    }


def main():
    rng = random.Random(SEED)
    windows = [random_window(rng) for _ in range(N_WINDOWS)]
    print(f"Ventanas (seed={SEED}, N={N_WINDOWS}, plazo={WINDOW_YEARS}y):")
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
                print(f"  {name:<18}  ret={m['return']:+8.2f}%  DD={m['maxdd']:+6.2f}%  "
                      f"PF={m['pf']:5.2f}  trades={m['trades']:3d}  hold={m['avg_hold']:>4.1f}d  "
                      f"({m['elapsed']:.1f}s)")
            except Exception as ex:
                print(f"  {name:<18}  FAIL · {ex}")
        sys.stdout.flush()
        print()

    aggs = {name: aggregate_with_ci(results[name], n_resamples=2000, seed=42)
            for name, _ in VARIANTS}
    print('═' * 100)
    print(f'AGREGADOS {N_WINDOWS} VENTANAS  (CIs 95% bootstrap)')
    print('═' * 100)
    print(format_aggregate_table(aggs, with_ci=True))
    print()

    # Gate Agresivo: cada SWING vs AGGR_PLUS_FAMOUS
    print('─── Gate Agresivo (cada SWING vs AGGR_PLUS_FAMOUS) ───')
    base_rows = results['AGGR_PLUS_FAMOUS']
    verdicts = {}
    for name, _ in VARIANTS:
        if name == 'AGGR_PLUS_FAMOUS':
            continue
        gate = aggressive_gate(base_rows, results[name], n_resamples=4000)
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

    # Bankruptcy guardrail extra: worst_DD ≤ −40% en CUALQUIER ventana = fail
    print('─── Tier 5 bankruptcy guardrail (worst_DD ≥ −40% en TODAS las ventanas) ───')
    for name, _ in VARIANTS:
        rows = [r[3] for r in results[name]]
        if not rows:
            continue
        worst = min(r['maxdd'] for r in rows)
        n_disaster = sum(1 for r in rows if r['maxdd'] < -40.0)
        status = '✓ SAFE' if n_disaster == 0 else f'✗ FAIL ({n_disaster}/{len(rows)} ventanas con DD < −40%)'
        print(f"  {name:<18}  worst_DD = {worst:+.2f}%   {status}")
    print()

    # Output JSON
    out_path = '/tmp/compare_swing_result.json'
    out = {
        'seed': SEED,
        'n_windows': N_WINDOWS,
        'window_years': WINDOW_YEARS,
        'variants': {name: extras for name, extras in VARIANTS},
        'aggregates': aggs,
        'verdicts': {name: {
            'accepted': v['accepted'],
            'verdict':  v['verdict'],
            'delta_sharpe': v['delta_sharpe'],
            'delta_return': v['delta_return'],
            'delta_maxdd':  v['delta_maxdd'],
        } for name, v in verdicts.items()},
        'rows': {name: [r[3] for r in results[name]] for name, _ in VARIANTS},
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"→ JSON: {out_path}")


if __name__ == '__main__':
    main()
