"""
Tier 5 Phase 5 — Tasa de acierto por timeframe con fechas aleatorias.

Mide, para cada variante × timeframe, la fracción de runs con retorno > 0
(tasa de acierto) y métricas auxiliares (avg_ret, win_rate de trades, alpha
vs B&H). Hecho con N fechas aleatorias por timeframe para tener significancia
estadística.

Por defecto:
  · 25 fechas aleatorias seed=42 (entre 2018-04 y 2024-04, ajustando por horizon)
  · 6 timeframes: 1W, 1M, 3M, 6M, 1Y, 2Y (los del Tier 5 + originales)
  · 3 variantes: AGGR_PLUS_FAMOUS (baseline Tier 4.2),
                 SWING_FAMOUS (Tier 5),
                 SWING_BROAD_TUNED (Tier 5 + Phase 4 sweep si está disponible)

Output: /tmp/win_rate_result.json + tabla per-timeframe + dashboard HTML.

Uso:
    PORT=5057 .venv/bin/python -u api.py &
    .venv/bin/python -u tools/measure_win_rate.py [N=25]
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from urllib import request as urlreq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.backtest_metrics import bootstrap_ci  # noqa: E402

API = os.environ.get('API_URL', 'http://localhost:5057')

EARLIEST = datetime(2018, 4, 1)
LATEST   = datetime(2024, 4, 1)

# Timeframes a medir — incluye plazos cortos del Tier 5
TIMEFRAMES = [
    ('1W',   7),
    ('1M',  30),
    ('3M',  90),
    ('6M', 180),
    ('1Y', 365),
    ('2Y', 730),
]

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

SWING = {**AGGR_PLUS,
         'mode': 'swing',
         'max_holding_days_swing': 14,
         'min_holding_days_swing': 1,
         'conf_floor': 0.50}

SWING_BROAD = {**SWING,
               'universe_mode': 'broad_random',
               'universe_seed': 42,
               'universe_size_stocks': 30,
               'universe_size_crypto': 20}


def load_swing_broad_tuned():
    """Si el sweep Phase 4 (sweep_swing_broad.py) corrió, aplicamos el ganador
    como variante SWING_BROAD_TUNED. Si no, usamos los defaults."""
    sweep_path = '/tmp/sweep_swing_broad_result.json'
    if not os.path.exists(sweep_path):
        return SWING_BROAD, 'broad_default(s70_k30)'
    try:
        sweep = json.load(open(sweep_path))
        best = sweep['best']
        tuned = {**SWING_BROAD,
                 'signal_percentile': best['signal_pct'],
                 'kelly_scale':       best['kelly_scale']}
        return tuned, f"broad_tuned({best['label']})"
    except Exception as ex:
        print(f"⚠ No pude leer sweep_swing_broad_result.json: {ex}; uso default.")
        return SWING_BROAD, 'broad_default(s70_k30)'


def call(start, end, body_extra, timeout=600):
    body = {**body_extra, 'start_date': start, 'end_date': end}
    req = urlreq.Request(
        f"{API}/api/paper/autonomous-backtest", method='POST',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
    )
    t0 = time.time()
    with urlreq.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    elapsed = time.time() - t0
    if payload.get('status') != 'success':
        raise RuntimeError(payload.get('error', '?'))
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


def gen_dates(n, max_horizon_days, seed):
    rng = random.Random(seed)
    span = (LATEST - EARLIEST).days - max_horizon_days
    seen = set()
    out = []
    while len(out) < n:
        d = rng.randint(0, max(1, span))
        if d in seen:
            continue
        seen.add(d)
        out.append((EARLIEST + timedelta(days=d)).strftime('%Y-%m-%d'))
    return sorted(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, nargs='?', default=25,
                        help='fechas aleatorias por timeframe (default 25)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='/tmp/win_rate_result.json')
    args = parser.parse_args()

    swing_broad_tuned, swing_broad_label = load_swing_broad_tuned()
    VARIANTS = [
        ('AGGR_PLUS_FAMOUS', AGGR_PLUS),
        ('SWING_FAMOUS',     SWING),
        ('SWING_BROAD',      swing_broad_tuned),
    ]

    # Para cada timeframe usamos N fechas aleatorias propias.
    # Las fechas para timeframes cortos pueden incluir más recientes (más espacio).
    by_tf_dates = {tf: gen_dates(args.N, days, args.seed + i)
                   for i, (tf, days) in enumerate(TIMEFRAMES)}

    total = sum(len(d) for d in by_tf_dates.values()) * len(VARIANTS)
    print(f"Win-rate measurement — {len(VARIANTS)} variants × {len(TIMEFRAMES)} TF "
          f"× {args.N} dates = {total} backtests")
    print(f"  Variants: {', '.join(name for name,_ in VARIANTS)}")
    print(f"  TFs:      {', '.join(tf for tf,_ in TIMEFRAMES)}")
    print(f"  SWING_BROAD config: {swing_broad_label}")
    print()

    results = {name: {tf: [] for tf, _ in TIMEFRAMES} for name, _ in VARIANTS}

    t0 = time.time()
    n_done = 0
    for tf, days in TIMEFRAMES:
        for sd in by_tf_dates[tf]:
            ed = (datetime.strptime(sd, '%Y-%m-%d') + timedelta(days=days)).strftime('%Y-%m-%d')
            for vname, vbody in VARIANTS:
                try:
                    r = call(sd, ed, vbody, timeout=600)
                    r['start'] = sd; r['end'] = ed; r['tf'] = tf
                    results[vname][tf].append(r)
                except Exception as ex:
                    print(f"  ✗ {vname} {sd} {tf}: {ex}")
                n_done += 1
            sys.stdout.flush()
        elapsed = time.time() - t0
        eta_min = (elapsed / max(n_done, 1)) * (total - n_done) / 60
        print(f"[{tf}] {len(by_tf_dates[tf])} dates done · "
              f"{n_done}/{total} · ETA {eta_min:.0f} min")

    elapsed_total = time.time() - t0
    print(f"\nTotal: {elapsed_total/60:.1f} min · {n_done}/{total} backtests")

    # ── Agregados por variante × tf ────────────────────────────────────
    summary = {}
    for vname, _ in VARIANTS:
        summary[vname] = {}
        for tf, _ in TIMEFRAMES:
            rs = results[vname][tf]
            if not rs:
                continue
            n      = len(rs)
            n_pos  = sum(1 for r in rs if r['return'] > 0)
            n_alpha= sum(1 for r in rs if r['alpha']  > 0)
            wr     = n_pos / n * 100
            wr_alpha = n_alpha / n * 100
            avg_ret  = sum(r['return'] for r in rs) / n
            med_ret  = sorted(r['return'] for r in rs)[n // 2]
            avg_alpha= sum(r['alpha']  for r in rs) / n
            avg_dd   = sum(r['maxdd']  for r in rs) / n
            worst_dd = min(r['maxdd']  for r in rs)
            avg_sh   = sum(r['sharpe'] for r in rs) / n
            avg_tr   = sum(r['trades'] for r in rs) / n
            ci_ret   = bootstrap_ci([r['return'] for r in rs])
            summary[vname][tf] = {
                'n': n, 'n_pos': n_pos, 'n_alpha': n_alpha,
                'win_rate_pct':  wr,
                'alpha_rate_pct': wr_alpha,
                'avg_ret':  avg_ret, 'med_ret': med_ret, 'ci95_ret': ci_ret,
                'avg_alpha': avg_alpha,
                'avg_dd':   avg_dd, 'worst_dd': worst_dd,
                'avg_sharpe': avg_sh, 'avg_trades': avg_tr,
            }

    # ── Tabla principal: tasa de acierto por TF × variante ─────────────
    print('\n' + '═' * 100)
    print('TASA DE ACIERTO (% ventanas con retorno > 0) — todas las variantes')
    print('═' * 100)
    header = f"{'TF':<4} " + ' '.join(f"{v:>22}" for v, _ in VARIANTS)
    print(header)
    print('─' * 100)
    for tf, _ in TIMEFRAMES:
        row = [f"{tf:<4}"]
        for vname, _ in VARIANTS:
            s = summary.get(vname, {}).get(tf)
            if not s:
                row.append(f"{'—':>22}")
                continue
            row.append(f"{s['win_rate_pct']:>5.1f}% ({s['n_pos']:>2}/{s['n']:<2}) "
                       f"avg{s['avg_ret']:+5.1f}%")
        print(' '.join(row))
    print('═' * 100)

    print('\n--- TASA DE ACIERTO vs B&H (% ventanas con alpha > 0) ---')
    print(header)
    print('─' * 100)
    for tf, _ in TIMEFRAMES:
        row = [f"{tf:<4}"]
        for vname, _ in VARIANTS:
            s = summary.get(vname, {}).get(tf)
            if not s:
                row.append(f"{'—':>22}")
                continue
            row.append(f"{s['alpha_rate_pct']:>5.1f}% ({s['n_alpha']:>2}/{s['n']:<2}) "
                       f"avg α{s['avg_alpha']:+5.1f}%")
        print(' '.join(row))

    print('\n--- worst_DD por TF (riesgo de cola) ---')
    print(header)
    print('─' * 100)
    for tf, _ in TIMEFRAMES:
        row = [f"{tf:<4}"]
        for vname, _ in VARIANTS:
            s = summary.get(vname, {}).get(tf)
            if not s:
                row.append(f"{'—':>22}")
                continue
            row.append(f"{'worst':>10}: {s['worst_dd']:+6.1f}%   ")
        print(' '.join(row))

    out = {
        'config': {
            'n_per_tf':   args.N,
            'seed':       args.seed,
            'timeframes': TIMEFRAMES,
            'variants':   {name: extras for name, extras in VARIANTS},
            'swing_broad_label': swing_broad_label,
            'started_at':  datetime.now().isoformat(),
            'elapsed_sec': round(elapsed_total, 1),
        },
        'summary': summary,
        'rows':    {vname: {tf: results[vname][tf] for tf, _ in TIMEFRAMES}
                    for vname, _ in VARIANTS},
        'dates_per_tf': by_tf_dates,
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n→ JSON: {args.out}")


if __name__ == '__main__':
    main()
