"""
run_random_grid.py — harness de backtest con fechas de inicio ALEATORIAS y
ventanas temporalmente disjuntas (dev vs lockbox) para una evaluación OOS honesta.

A diferencia de run_performance_grid.py (fechas de inicio fijas, escogidas a mano
y por tanto susceptibles de cherry-picking), aquí cada horizonte muestrea N fechas
de inicio con un RNG seedeado y reproducible. window_idx es estable entre variantes
(mismo seed) → habilita A/B pareado posterior vía delta_ci.

Universos DISJUNTOS en el tiempo (clave para que el OOS sea honesto):
  · dev:     2014-01-01 .. ventanas que terminan ≤ 2023-12-31
  · lockbox: starts en 2024-2025, ventanas usando datos hasta hoy

Reusa VARIANT_CONFIGS de run_performance_grid.py (no redefine variantes) y las
métricas de utils.backtest_metrics. Registra UNA prueba en el trial_registry por
ejecución (alimenta el haircut por multiple-testing del Deflated Sharpe Ratio).

DSR aquí es una APROXIMACIÓN: usa la varianza del Sharpe entre ventanas como
sharpe_variance y el Sharpe por ventana (anualizado por el backtester) como
observación. No es el DSR canónico sobre retornos diarios de una sola serie.

Uso:
    PORT=5057 .venv/bin/python -u api.py &
    PYTHONPATH=. .venv/bin/python -u tools/run_random_grid.py --variant faber_qqq --universe dev --n-windows 30 --seed 42
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from urllib import request as urlreq

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.run_performance_grid import API, VARIANT_CONFIGS
from utils.backtest_metrics import (
    build_horizon_summary,
    deflated_sharpe_ratio,
    format_horizon_table,
    sharpe_stats_from_rows,
)
from utils.trial_registry import count_trials, register_trial

HORIZONS = [
    ('2W', 14),
    ('1M', 30),
    ('3M', 91),
    ('6M', 182),
    ('1Y', 365),
    ('2Y', 730),
]

BEAT_KEYS = ('alpha_vs_spy_pct', 'alpha_vs_qqq_pct', 'alpha_vs_6040_pct')


def universe_bounds(universe: str, horizon_days: int, today):
    if universe == 'dev':
        lo = datetime(2014, 1, 1).date()
        end_cap = datetime(2023, 12, 31).date()
        latest_start = end_cap - timedelta(days=horizon_days)
    else:
        lo = datetime(2024, 1, 1).date()
        start_cap = datetime(2025, 12, 31).date()
        latest_start = min(start_cap, today - timedelta(days=horizon_days))
    return lo, latest_start


def sample_starts(universe: str, horizon_days: int, n: int, seed: int, h_index: int, today):
    lo, latest_start = universe_bounds(universe, horizon_days, today)
    span = (latest_start - lo).days
    if span <= 0:
        return [], span, 0
    rng = np.random.default_rng(seed + h_index)
    offsets = rng.integers(0, span + 1, size=n * 4)
    seen = []
    seen_set = set()
    for off in offsets:
        d = lo + timedelta(days=int(off))
        if d not in seen_set:
            seen_set.add(d)
            seen.append(d)
        if len(seen) >= n:
            break
    return seen[:n], span, len(seen[:n])


def call(start_date: str, end_date: str, body_base: dict) -> dict:
    body = {**body_base, 'start_date': start_date, 'end_date': end_date}
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
        raise RuntimeError(f"{start_date}→{end_date} → {payload.get('error', payload)}")
    p = payload['performance']
    ret = p.get('total_return_pct', 0.0)
    bh = p.get('buy_hold_return_pct', 0.0)
    return {
        'start_date': start_date,
        'end_date': end_date,
        'return_pct': ret,
        'bh_return_pct': bh,
        'alpha_pct': ret - bh,
        'maxdd_pct': p.get('max_drawdown_pct', 0.0),
        'sharpe': p.get('sharpe_ratio', 0.0),
        'sortino': p.get('sortino_ratio', 0.0),
        'profit_factor': p.get('profit_factor', 0.0),
        'win_rate_pct': p.get('win_rate_pct', 0.0),
        'total_trades': p.get('total_trades', 0),
        'spy_return_pct': p.get('spy_return_pct', None),
        'qqq_return_pct': p.get('qqq_return_pct', None),
        'sixtyforty_return_pct': p.get('sixtyforty_return_pct', None),
        'alpha_vs_spy_pct': p.get('alpha_vs_spy_pct', None),
        'alpha_vs_qqq_pct': p.get('alpha_vs_qqq_pct', None),
        'alpha_vs_6040_pct': p.get('alpha_vs_6040_pct', None),
        'elapsed': elapsed,
    }


def rows_from_runs(runs):
    rows = []
    for r in runs:
        rows.append((
            r['window_idx'],
            r['start_date'],
            r['end_date'],
            {
                'return': r.get('return_pct'),
                'bh': r.get('bh_return_pct'),
                'alpha': r.get('alpha_pct'),
                'maxdd': r.get('maxdd_pct'),
                'sharpe': r.get('sharpe'),
                'sortino': r.get('sortino'),
                'pf': r.get('profit_factor'),
                'wr': r.get('win_rate_pct'),
                'trades': r.get('total_trades'),
            },
        ))
    return rows


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    return obj


def out_path_for(variant_key: str, universe: str) -> str:
    base = VARIANT_CONFIGS[variant_key][2]
    root, ext = os.path.splitext(base)
    return f"{root}_random_{universe}{ext}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=list(VARIANT_CONFIGS.keys()), default='faber_qqq')
    parser.add_argument('--universe', choices=['dev', 'lockbox'], default='dev')
    parser.add_argument('--n-windows', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    variant_label, body_base, _ = VARIANT_CONFIGS[args.variant]
    out_path = out_path_for(args.variant, args.universe)
    today = datetime.now().date()

    n_trials = register_trial(variant_label, body_base, args.universe)

    print(f"Random grid · variant={variant_label} · universe={args.universe} · "
          f"n_windows={args.n_windows} · seed={args.seed} · today={today}")
    print(f"config={body_base}")
    print(f"trial_registry distinct trials = {n_trials}")
    print()

    runs = []
    skipped = []
    feasible_horizons = []

    for h_index, (h_label, h_days) in enumerate(HORIZONS):
        starts, span, got = sample_starts(args.universe, h_days, args.n_windows, args.seed, h_index, today)
        if span <= 0:
            msg = (f"INFEASIBLE: {h_label} ({h_days}d) in universe {args.universe} "
                   f"(span={span}d ≤ 0; horizon does not fit) — skipping")
            print(msg)
            skipped.append({'horizon': h_label, 'reason': msg})
            continue
        if got < args.n_windows:
            print(f"  NOTE {h_label}: requested {args.n_windows} distinct starts, "
                  f"obtained {got} (span={span}d) — no silent truncation")
        feasible_horizons.append((h_label, h_days))
        for window_idx, sd_dt in enumerate(starts):
            sd = sd_dt.strftime('%Y-%m-%d')
            ed = (sd_dt + timedelta(days=h_days)).strftime('%Y-%m-%d')
            try:
                r = call(sd, ed, body_base)
                r['horizon'] = h_label
                r['horizon_days'] = h_days
                r['window_idx'] = window_idx
                runs.append(r)
                print(f"  {h_label} #{window_idx:>2} {sd}→{ed}: "
                      f"ret={r['return_pct']:+7.2f}%  bh={r['bh_return_pct']:+7.2f}%  "
                      f"alpha={r['alpha_pct']:+7.2f}%  DD={r['maxdd_pct']:+6.2f}%  "
                      f"trades={r['total_trades']:3d}  ({r['elapsed']:.1f}s)")
            except Exception as ex:
                print(f"  {h_label} #{window_idx:>2} {sd}→{ed}: FAIL · {ex}")
                skipped.append({'horizon': h_label, 'start_date': sd, 'end_date': ed,
                                'window_idx': window_idx, 'reason': str(ex)[:200]})

    by_h = {}
    for r in runs:
        by_h.setdefault(r['horizon'], []).append(r)

    summaries = {}
    summary_by_horizon = {}
    for h_label, _ in HORIZONS:
        rs = by_h.get(h_label, [])
        if not rs:
            continue
        s = build_horizon_summary(rs, BEAT_KEYS)
        summaries[h_label] = s
        summary_by_horizon[h_label] = json_safe(s)

    print()
    print('=' * 130)
    print('AGGREGATES BY HORIZON (random starts, bootstrap CI 95%)')
    print('=' * 130)
    if summaries:
        print(format_horizon_table(summaries))
    else:
        print('  (no runs)')

    print()
    print('DEFLATED SHARPE RATIO (approx: sharpe_variance = cross-window Sharpe variance; '
          'observation = annualized per-window Sharpe)')
    dsr_by_horizon = {}
    nt = count_trials()
    for h_label, _ in HORIZONS:
        rs = by_h.get(h_label, [])
        if not rs:
            continue
        rows = rows_from_runs(rs)
        mean_sh, var_sh, skw, krt, n = sharpe_stats_from_rows(rows)
        dsr = deflated_sharpe_ratio(mean_sh, nt, var_sh, skw, krt, n)
        dsr_by_horizon[h_label] = float(dsr)
        print(f"  {h_label}: DSR={dsr:.4f} (n_trials={nt}, n_obs={n}, "
              f"mean_Sharpe={mean_sh:+.3f}, var_Sharpe={var_sh:.4f})")

    out = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'variant': variant_label,
        'universe': args.universe,
        'seed': args.seed,
        'n_windows': args.n_windows,
        'today': str(today),
        'horizons': [[l, d] for l, d in HORIZONS],
        'runs': runs,
        'skipped': skipped,
        'summary_by_horizon': summary_by_horizon,
        'dsr_by_horizon': dsr_by_horizon,
        'n_trials': nt,
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n→ JSON: {out_path}")


if __name__ == '__main__':
    main()
