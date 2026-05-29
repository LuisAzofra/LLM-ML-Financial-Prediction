"""
Tier 5 — Test cuantitativo de sesgo de supervivencia.

Corre la grid (fechas × plazos) del bot AGGR_PLUS sobre dos universos:
  · 'famous'        — 6 tech famosas + 4 cryptos top (universo histórico)
  · 'broad_random'  — 30 acciones random S&P 500 + 20 cryptos top, 3 seeds

Computa, por cada (fecha, plazo, universo, [seed]):
  · total_return_pct, buy_hold_return_pct, max_drawdown_pct, sharpe, trades

Y compara famous vs broad agregado por plazo:
  · Δ_avg_return        (paired bootstrap CI95 sobre fechas)
  · Δ_alpha_vs_BH       (idem)
  · Δ_max_drawdown      (idem)

Output: /tmp/survivorship_bias_result.json + tabla en stdout.

Uso:
    PORT=5057 .venv/bin/python -u api.py &
    .venv/bin/python -u tools/measure_survivorship_bias.py             # default 3 seeds
    .venv/bin/python -u tools/measure_survivorship_bias.py --quick     # 4 fechas, 1 seed
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from urllib import request as urlreq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.backtest_metrics import bootstrap_ci  # noqa: E402

API = os.environ.get('API_URL', 'http://localhost:5057')

# Reutilizamos las START_DATES + HORIZONS del run_performance_grid existente
# para que la comparación sea side-by-side con datos previos.
HORIZONS = [
    ('3M',   90),
    ('6M',  180),
    ('1Y',  365),
    ('2Y',  730),
]

START_DATES = [
    '2018-04-02',
    '2019-01-02',
    '2019-07-01',
    '2020-01-02',
    '2020-07-01',
    '2021-01-04',
    '2021-07-01',
    '2022-01-03',
    '2022-07-01',
    '2023-01-03',
]

QUICK_DATES = ['2019-01-02', '2020-07-01', '2022-01-03', '2023-01-03']

# Variante AGGR_PLUS validada Tier 4.2 — fija para el test del sesgo.
AGGR_PLUS_BASE = {
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
}


def call(start_date: str, end_date: str, body_extra: dict, timeout: int = 600) -> dict:
    body = {**AGGR_PLUS_BASE, **body_extra,
            'start_date': start_date, 'end_date': end_date}
    req = urlreq.Request(
        f"{API}/api/paper/autonomous-backtest",
        method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urlreq.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    elapsed = time.time() - t0
    if payload.get('status') != 'success':
        raise RuntimeError(f"{start_date}→{end_date} → {payload.get('error', payload)}")
    p = payload['performance']
    cfg_universe = payload.get('config', {}).get('universe', {})
    return {
        'start_date':    start_date,
        'end_date':      end_date,
        'return_pct':    p.get('total_return_pct', 0.0),
        'bh_return_pct': p.get('buy_hold_return_pct', 0.0),
        'alpha_pct':     p.get('total_return_pct', 0.0) - p.get('buy_hold_return_pct', 0.0),
        'maxdd_pct':     p.get('max_drawdown_pct', 0.0),
        'sharpe':        p.get('sharpe_ratio', 0.0),
        'profit_factor': p.get('profit_factor', 0.0),
        'win_rate_pct':  p.get('win_rate_pct', 0.0),
        'total_trades':  p.get('total_trades', 0),
        'universe':      cfg_universe,
        'elapsed':       elapsed,
    }


def add_days(date_str: str, n: int) -> str:
    return (datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=n)).strftime('%Y-%m-%d')


def paired_delta_ci(famous_rows: list, broad_rows: list, key: str,
                    n_resamples: int = 1000, seed: int = 7) -> dict:
    """Paired bootstrap por fecha: para cada fecha tenemos 1 punto famous vs
    avg(broad seeds). El delta es broad − famous (positivo = broad mejor)."""
    pairs = []
    by_date_famous = {r['start_date']: r for r in famous_rows}
    by_date_broad: dict = {}
    for r in broad_rows:
        by_date_broad.setdefault(r['start_date'], []).append(r)
    for d, fr in by_date_famous.items():
        if d not in by_date_broad:
            continue
        broad_avg = sum(b[key] for b in by_date_broad[d]) / len(by_date_broad[d])
        pairs.append(broad_avg - fr[key])
    if not pairs:
        return {'point': 0.0, 'lo': 0.0, 'hi': 0.0, 'n_pairs': 0}
    point, lo, hi = bootstrap_ci(pairs, n_resamples=n_resamples, seed=seed)
    return {'point': round(point, 3), 'lo': round(lo, 3), 'hi': round(hi, 3),
            'n_pairs': len(pairs)}


def aggregate(rows: list, key: str = 'return_pct') -> dict:
    if not rows:
        return {'mean': 0.0, 'lo': 0.0, 'hi': 0.0, 'n': 0}
    vals = [r[key] for r in rows]
    point, lo, hi = bootstrap_ci(vals)
    return {'mean': round(point, 2), 'lo': round(lo, 2), 'hi': round(hi, 2),
            'n': len(vals)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help="4 fechas × 1 seed (~15 min en lugar de ~2h)")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 7, 123],
                        help="seeds del sample broad_random (default 42 7 123)")
    parser.add_argument('--n_stocks', type=int, default=30)
    parser.add_argument('--n_crypto', type=int, default=20)
    parser.add_argument('--out', type=str, default='/tmp/survivorship_bias_result.json')
    args = parser.parse_args()

    dates = QUICK_DATES if args.quick else START_DATES
    seeds = args.seeds[:1] if args.quick else args.seeds

    print(f"[survivorship] dates={len(dates)} horizons={len(HORIZONS)} "
          f"seeds={seeds} mode={'quick' if args.quick else 'full'}")
    print(f"[survivorship] runs estimados: famous={len(dates)*len(HORIZONS)}, "
          f"broad={len(dates)*len(HORIZONS)*len(seeds)}")

    results = {
        'famous':      [],
        'broad':       [],
        'config': {
            'aggr_plus_base': AGGR_PLUS_BASE,
            'horizons':       HORIZONS,
            'start_dates':    dates,
            'seeds':          seeds,
            'n_stocks':       args.n_stocks,
            'n_crypto':       args.n_crypto,
            'started_at':     datetime.now().isoformat(),
        },
    }

    t_global = time.time()

    # 1) FAMOUS (default universe_mode='famous')
    for sd in dates:
        for hname, hd in HORIZONS:
            ed = add_days(sd, hd)
            try:
                r = call(sd, ed, {'universe_mode': 'famous',
                                   'include_etfs': True})  # AGGR_PLUS = + ETFs
                r['horizon'] = hname
                r['horizon_days'] = hd
                results['famous'].append(r)
                print(f"  famous  {sd} {hname:>3}  ret={r['return_pct']:+7.2f}%  "
                      f"bh={r['bh_return_pct']:+7.2f}%  dd={r['maxdd_pct']:+7.2f}%  "
                      f"trades={r['total_trades']:>3}  ({r['elapsed']:.0f}s)")
            except Exception as e:
                print(f"  famous  {sd} {hname:>3}  ERROR: {e}")
                results['famous'].append({'start_date': sd, 'end_date': ed,
                                          'horizon': hname, 'error': str(e)})
            sys.stdout.flush()

    # 2) BROAD (3 seeds × 30 stocks + 20 crypto)
    for seed in seeds:
        for sd in dates:
            for hname, hd in HORIZONS:
                ed = add_days(sd, hd)
                try:
                    r = call(sd, ed, {'universe_mode': 'broad_random',
                                       'universe_seed': seed,
                                       'universe_size_stocks': args.n_stocks,
                                       'universe_size_crypto': args.n_crypto})
                    r['horizon'] = hname
                    r['horizon_days'] = hd
                    r['seed'] = seed
                    results['broad'].append(r)
                    print(f"  broad   {sd} {hname:>3} seed={seed:>3}  "
                          f"ret={r['return_pct']:+7.2f}%  bh={r['bh_return_pct']:+7.2f}%  "
                          f"dd={r['maxdd_pct']:+7.2f}%  trades={r['total_trades']:>3}  "
                          f"({r['elapsed']:.0f}s)")
                except Exception as e:
                    print(f"  broad   {sd} {hname:>3} seed={seed:>3}  ERROR: {e}")
                    results['broad'].append({'start_date': sd, 'end_date': ed,
                                              'horizon': hname, 'seed': seed,
                                              'error': str(e)})
                sys.stdout.flush()

    elapsed_total = time.time() - t_global
    print(f"\n[survivorship] terminado en {elapsed_total/60:.1f} min")

    # 3) Agregaciones por plazo
    summary = {}
    for hname, _ in HORIZONS:
        f_h = [r for r in results['famous'] if r.get('horizon') == hname and 'error' not in r]
        b_h = [r for r in results['broad']  if r.get('horizon') == hname and 'error' not in r]
        if not f_h or not b_h:
            continue
        summary[hname] = {
            'famous_avg_return':  aggregate(f_h, 'return_pct'),
            'broad_avg_return':   aggregate(b_h, 'return_pct'),
            'famous_avg_alpha':   aggregate(f_h, 'alpha_pct'),
            'broad_avg_alpha':    aggregate(b_h, 'alpha_pct'),
            'famous_worst_dd':    min((r['maxdd_pct'] for r in f_h), default=0.0),
            'broad_worst_dd':     min((r['maxdd_pct'] for r in b_h), default=0.0),
            'delta_return_paired': paired_delta_ci(f_h, b_h, 'return_pct'),
            'delta_alpha_paired':  paired_delta_ci(f_h, b_h, 'alpha_pct'),
            'delta_maxdd_paired':  paired_delta_ci(f_h, b_h, 'maxdd_pct'),
        }

    results['summary'] = summary
    results['config']['finished_at'] = datetime.now().isoformat()
    results['config']['elapsed_sec'] = round(elapsed_total, 1)

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[survivorship] JSON: {args.out}")

    # 4) Tabla resumen
    print("\n" + "=" * 95)
    print(f"{'Plazo':<5} {'famous_ret':>13} {'broad_ret':>13} "
          f"{'Δ_ret CI95':>22} {'Δ_alpha CI95':>22}")
    print("-" * 95)
    for hname, s in summary.items():
        f_ret = s['famous_avg_return']
        b_ret = s['broad_avg_return']
        d_ret = s['delta_return_paired']
        d_alp = s['delta_alpha_paired']
        f_str = f"{f_ret['mean']:+6.2f}% [{f_ret['lo']:+5.1f},{f_ret['hi']:+5.1f}]"
        b_str = f"{b_ret['mean']:+6.2f}% [{b_ret['lo']:+5.1f},{b_ret['hi']:+5.1f}]"
        dr_str = f"{d_ret['point']:+6.2f}pp [{d_ret['lo']:+5.1f},{d_ret['hi']:+5.1f}]"
        da_str = f"{d_alp['point']:+6.2f}pp [{d_alp['lo']:+5.1f},{d_alp['hi']:+5.1f}]"
        print(f"{hname:<5} {f_str:>13} {b_str:>13} {dr_str:>22} {da_str:>22}")
    print("=" * 95)

    # 5) Veredicto del gate de Fase 1
    print("\nGate Fase 1: Δ_avg_return(broad − famous) IC95% lo > −10pp por plazo")
    failed = []
    for hname, s in summary.items():
        d = s['delta_return_paired']
        passed = d['lo'] > -10.0
        status = "✓ PASA" if passed else "✗ FALLA (Fase 4 obligatoria)"
        print(f"  {hname}: Δ_ret CI lo = {d['lo']:+.2f}pp  {status}")
        if not passed:
            failed.append(hname)
    if failed:
        print(f"\n⚠ Sesgo de supervivencia confirmado en {failed} → Fase 4 (recalibración)")
    else:
        print(f"\n✓ Todos los plazos pasan: el bot tiene edge real (no es sólo selección de famosas)")


if __name__ == '__main__':
    main()
