"""
Tier 4.1 — Fase 0: medir cuántas señales rechaza cada filtro de entrada.

Ejecuta el backtest con `debug_filter_counts=true` sobre las 6 ventanas
seed=42 y reporta una tabla con el % de candidatos rechazados por cada
filtro. Permite priorizar dónde aflojar.
"""
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from urllib import request as urlreq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API = "http://localhost:5057"
N_WINDOWS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
WINDOW_YEARS = 2
SEED = 42

EARLIEST = datetime(2018, 1, 1)
LATEST   = datetime(2024, 4, 1)
DELTA_DAYS = (LATEST - EARLIEST).days

MED_BASE = {
    'mode': 'trend',
    'initial_capital': 100_000,
    'allow_short': False,
    'signal_percentile': 0.82,
    'kelly_scale': 0.22,
    'trend_reverse_exit': True,
    'disable_atr_stop': True,
    'max_holding_days': 120,
    'target_vol': 0.20,
    'use_llm': False,
    'debug_filter_counts': True,
}


def random_window(rng):
    days = rng.randint(0, DELTA_DAYS)
    start = EARLIEST + timedelta(days=days)
    end   = start + timedelta(days=365 * WINDOW_YEARS)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def call(start, end):
    body = {**MED_BASE, 'start_date': start, 'end_date': end}
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
        raise RuntimeError(f"{start}→{end} → {payload.get('error', payload)}")
    return payload, elapsed


def main():
    rng = random.Random(SEED)
    windows = [random_window(rng) for _ in range(N_WINDOWS)]
    print(f"Midiendo rechazos por filtro sobre {N_WINDOWS} ventanas (seed={SEED}, MED_NOLLM):")
    print()

    all_counts = []
    totals = {}
    for i, (s, e) in enumerate(windows, 1):
        try:
            payload, elapsed = call(s, e)
            fc = payload.get('filter_counts') or {}
            n_total = fc.get('total_evaluated', 0)
            ntrades = payload.get('performance', {}).get('total_trades', 0)
            print(f"  W{i}  {s}→{e}  total_evaluated={n_total:6d}  trades={ntrades:3d}  ({elapsed:.1f}s)")
            all_counts.append((i, s, e, fc))
            for k, v in fc.items():
                totals[k] = totals.get(k, 0) + v
        except Exception as ex:
            print(f"  W{i}  {s}→{e}  FAIL · {ex}")

    if not all_counts:
        print("Sin datos.")
        return

    print()
    print('═' * 92)
    print('RECHAZOS POR FILTRO (agregado sobre las 6 ventanas)')
    print('═' * 92)
    total_evaluated = totals.get('total_evaluated', 0)
    if total_evaluated == 0:
        print("total_evaluated=0 — el endpoint no expuso filter_counts. Reinicia el API.")
        return
    print(f"  {'filtro':<20}  {'rechazados':>11}  {'% del total':>12}")
    # ordenado por % de rechazo desc
    filt_keys = ['signal_percentile', 'quality_gate', 'asset_r2_gate',
                 'no_direction', 'sma200_filter', 'golden_cross',
                 'adx_filter', 'crash_filter', 'spy_regime']
    rows = []
    for k in filt_keys:
        n = totals.get(k, 0)
        pct = n / total_evaluated * 100
        rows.append((k, n, pct))
    rows.sort(key=lambda r: r[1], reverse=True)
    for k, n, pct in rows:
        bar = '█' * int(pct / 2)   # 1 char = 2%
        print(f"  {k:<20}  {n:>11d}  {pct:>10.2f} %  {bar}")
    n_rejected = sum(n for _, n, _ in rows)
    n_passed = total_evaluated - n_rejected
    print(f"  {'-'*20}  {'-'*11}  {'-'*12}")
    print(f"  {'TOTAL evaluated':<20}  {total_evaluated:>11d}  {'100.00':>10} %")
    print(f"  {'pasaron todos':<20}  {n_passed:>11d}  {n_passed/total_evaluated*100:>10.2f} %")

    print()
    print('─── Detalle por ventana ───')
    print(f"  {'window':<28}  {'eval':>7}  " + '  '.join(f"{k[:8]:>8}" for k in filt_keys))
    for i, s, e, fc in all_counts:
        row = f"  W{i} {s[:7]}..{e[2:7]}".ljust(30) + f"  {fc.get('total_evaluated',0):>7d}  "
        row += '  '.join(f"{fc.get(k, 0):>8d}" for k in filt_keys)
        print(row)

    out_path = '/tmp/filter_rejection_result.json'
    with open(out_path, 'w') as f:
        json.dump({
            'seed': SEED, 'n_windows': N_WINDOWS,
            'windows': windows, 'totals': totals,
            'detail': [{'window': i, 'start': s, 'end': e, 'counts': fc}
                       for i, s, e, fc in all_counts],
        }, f, indent=2)
    print(f"\n→ JSON: {out_path}")


if __name__ == '__main__':
    main()
