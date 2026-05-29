"""
Ejecuta una cuadrícula (fecha_inicio × plazo) de backtests sobre el bot
para responder ¿es rentable? sin sesgar por una sola ventana.

Plazos: 3M, 6M, 1Y, 2Y.
Fechas de inicio: cubren bull / bear / lateral entre 2018-04 y 2023-04.

Variantes soportadas (`--variant`):
  · MED  (default)   — kelly_scale=0.22, signal_percentile=0.82  (legacy)
  · AGGR             — kelly_scale=0.30, signal_percentile=0.70  (Tier 4.1)

Salida JSON con suffix `_<variant>`. Uso:
    PORT=5057 .venv/bin/python -u api.py &
    .venv/bin/python -u tools/run_performance_grid.py            # MED (default, sobreescribe /tmp/perf_grid_result.json)
    .venv/bin/python -u tools/run_performance_grid.py --variant aggr   # AGGR → /tmp/perf_grid_aggr_result.json
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from urllib import request as urlreq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API = "http://localhost:5057"

# Plazos en meses → días aproximados
HORIZONS = [
    ('3M', 90),
    ('6M', 180),
    ('1Y', 365),
    ('2Y', 730),
]

# Tier 5: plazos cortos para evaluar la variante SWING. Usados solo cuando
# `--variant swing` o `--include-short`. NO se mezclan con HORIZONS por
# defecto para no inflar runtime de los grids existentes (40 → 80 backtests).
HORIZONS_SHORT = [
    ('1W',  7),
    ('2W', 14),
    ('1M', 30),
    ('2M', 60),
]

# Fechas representativas: 2 puntos en cada año entre 2018-2023.
# Cubre: pre-COVID (2019), crash (2020-Q1), bull recovery (2020-Q3),
# bull peak (2021), bear (2022), recovery (2023).
START_DATES = [
    '2018-04-02',  # pre-trade-war volatility
    '2019-01-02',  # bull recovery 2019
    '2019-07-01',  # mid-bull
    '2020-01-02',  # pre-COVID
    '2020-07-01',  # post-COVID rebound
    '2021-01-04',  # bull peak inicio
    '2021-07-01',  # mid-bull peak
    '2022-01-03',  # comienzo bear
    '2022-07-01',  # mid-bear
    '2023-01-03',  # recovery temprana
]

# Configuración base — variante MED por consistencia con Tiers anteriores
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
}

# Tier 4.1 — variante AGGR validada en compare_aggressive.py 12 ventanas
AGGR_BASE = {**MED_BASE, 'signal_percentile': 0.70, 'kelly_scale': 0.30}

# Tier 4.2 — AGGR_PLUS = AGGR + C3 top-N rotation con ETFs como pool extendido,
# validado en compare_candidates.py --cand C3 --n_windows 24:
# Δ_Sharpe +0.181 [+0.119,+0.247] sig, Δ_return +35.34pp [+9.85,+62.99] sig.
AGGR_PLUS = {**AGGR_BASE,
             'include_etfs': True,        # universo a 15 (10 + 5 ETFs)
             'topn_rotation': True,       # ranking por momentum 60d
             'topn_value': 5,
             'max_concurrent': 5}

VARIANT_CONFIGS = {
    'med':       ('MED_NOLLM',         MED_BASE,  '/tmp/perf_grid_result.json'),
    'aggr':      ('AGGR_KELLY_PCT',    AGGR_BASE, '/tmp/perf_grid_aggr_result.json'),
    'aggr_plus': ('AGGR_PLUS_C3',      AGGR_PLUS, '/tmp/perf_grid_aggr_plus_result.json'),
}

# Tier 5: variante SWING (mode='swing', holding 1-14d, ATR stop ON, conf
# floor exit). Hereda AGGR_PLUS (signal_pct=0.70, kelly=0.30, ETFs, topN
# rotation) y le añade los flags swing.
SWING_BASE = {**AGGR_PLUS,
              'mode': 'swing',
              'max_holding_days_swing': 14,
              'min_holding_days_swing': 1,
              'conf_floor': 0.50}
# AMPLIO: SWING + universo broad_random (test sesgo + corto plazo a la vez)
SWING_BROAD_42 = {**SWING_BASE,
                  'universe_mode': 'broad_random',
                  'universe_seed': 42,
                  'universe_size_stocks': 30,
                  'universe_size_crypto': 20}

VARIANT_CONFIGS['swing']       = ('SWING_DYNAMIC',  SWING_BASE,
                                  '/tmp/perf_grid_swing_result.json')
VARIANT_CONFIGS['swing_broad'] = ('SWING_BROAD_42', SWING_BROAD_42,
                                  '/tmp/perf_grid_swing_broad_result.json')


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
    return {
        'start_date': start_date,
        'end_date': end_date,
        'return_pct':    p.get('total_return_pct', 0.0),
        'bh_return_pct': p.get('buy_hold_return_pct', 0.0),
        'alpha_pct':     p.get('total_return_pct', 0.0) - p.get('buy_hold_return_pct', 0.0),
        'maxdd_pct':     p.get('max_drawdown_pct', 0.0),
        'sharpe':        p.get('sharpe_ratio', 0.0),
        'sortino':       p.get('sortino_ratio', 0.0),
        'profit_factor': p.get('profit_factor', 0.0),
        'win_rate_pct':  p.get('win_rate_pct', 0.0),
        'total_trades':  p.get('total_trades', 0),
        'elapsed':       elapsed,
    }


def add_days(date_str: str, n: int) -> str:
    return (datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=n)).strftime('%Y-%m-%d')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=list(VARIANT_CONFIGS.keys()), default='med',
                        help='med (default, baseline), aggr (Tier 4.1), aggr_plus (Tier 4.2), '
                             'swing (Tier 5), swing_broad (Tier 5 + universo broad)')
    parser.add_argument('--include-short', action='store_true',
                        help='Tier 5: añade plazos cortos (1W/2W/1M/2M) a la grid. '
                             'Auto-on para variantes swing*.')
    args = parser.parse_args()
    variant_label, body_base, out_path = VARIANT_CONFIGS[args.variant]

    # Tier 5: swing y swing_broad usan automáticamente plazos cortos. Cualquier
    # variante puede pedirlos con --include-short.
    use_short = args.include_short or args.variant.startswith('swing')
    horizons_eff = (HORIZONS_SHORT + HORIZONS) if use_short else list(HORIZONS)

    today = datetime.now().date()
    runs = []
    skipped = []

    print(f"Grid: {len(START_DATES)} fechas × {len(horizons_eff)} plazos = "
          f"{len(START_DATES)*len(horizons_eff)} backtests "
          f"({'incl. cortos' if use_short else 'plazos largos'})")
    print(f"Variante: {variant_label}  config={body_base}")
    print()

    for sd in START_DATES:
        sd_dt = datetime.strptime(sd, '%Y-%m-%d').date()
        for h_label, h_days in horizons_eff:
            end_dt = sd_dt + timedelta(days=h_days)
            if end_dt > today:
                skipped.append((sd, h_label, 'fin futuro'))
                continue
            ed = end_dt.strftime('%Y-%m-%d')
            try:
                r = call(sd, ed, body_base)
                r['horizon'] = h_label
                r['horizon_days'] = h_days
                runs.append(r)
                print(f"  {sd} ({h_label}): ret={r['return_pct']:+7.2f}%  bh={r['bh_return_pct']:+7.2f}%  alpha={r['alpha_pct']:+7.2f}%  DD={r['maxdd_pct']:+6.2f}%  trades={r['total_trades']:3d}  ({r['elapsed']:.1f}s)")
            except Exception as ex:
                print(f"  {sd} ({h_label}): FAIL · {ex}")
                skipped.append((sd, h_label, str(ex)[:80]))

    # Agregados por plazo
    print()
    print('═' * 120)
    print('AGREGADOS POR PLAZO')
    print('═' * 120)
    print(f"  {'plazo':<6}  {'n':>3}  {'avg_ret':>9}  {'med_ret':>9}  {'avg_alpha':>10}  {'avg_DD':>8}  {'avg_Sharpe':>10}  {'win_rate':>9}  {'%pos':>6}")
    by_h = {}
    for r in runs:
        by_h.setdefault(r['horizon'], []).append(r)

    summary_by_horizon = []
    for h_label, _ in horizons_eff:
        rs = by_h.get(h_label, [])
        if not rs:
            continue
        n = len(rs)
        rets   = sorted(r['return_pct']   for r in rs)
        alphas = sorted(r['alpha_pct']    for r in rs)
        dds    = sorted(r['maxdd_pct']    for r in rs)
        sharps = [r['sharpe'] for r in rs]
        wrs    = [r['win_rate_pct'] for r in rs]
        pos    = sum(1 for r in rs if r['return_pct'] > 0) / n * 100
        avg_ret    = sum(rets) / n
        med_ret    = rets[n // 2]
        avg_alpha  = sum(alphas) / n
        avg_dd     = sum(dds) / n
        avg_sharpe = sum(sharps) / n
        avg_wr     = sum(wrs) / n
        summary_by_horizon.append({
            'horizon': h_label, 'n': n,
            'avg_return': avg_ret, 'median_return': med_ret,
            'avg_alpha': avg_alpha, 'avg_maxdd': avg_dd,
            'avg_sharpe': avg_sharpe, 'avg_win_rate': avg_wr,
            'pct_positive': pos,
        })
        print(f"  {h_label:<6}  {n:>3}  {avg_ret:+8.2f}%  {med_ret:+8.2f}%  {avg_alpha:+9.2f}%  {avg_dd:+7.2f}%  {avg_sharpe:+9.2f}  {avg_wr:>7.1f}%  {pos:>5.1f}%")

    # Resumen global
    if runs:
        print()
        all_rets = [r['return_pct'] for r in runs]
        all_alphas = [r['alpha_pct'] for r in runs]
        n_pos = sum(1 for r in runs if r['return_pct'] > 0)
        n_alpha = sum(1 for r in runs if r['alpha_pct'] > 0)
        print(f"GLOBAL ({len(runs)} runs):")
        print(f"  avg_return   = {sum(all_rets)/len(runs):+.2f}%")
        print(f"  pct_positive = {n_pos}/{len(runs)} = {n_pos/len(runs)*100:.1f}%")
        print(f"  pct_alpha+   = {n_alpha}/{len(runs)} = {n_alpha/len(runs)*100:.1f}%")
        print(f"  median_alpha = {sorted(all_alphas)[len(runs)//2]:+.2f}%")

    out = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'variant': variant_label,
        'horizons': [(l, d) for l, d in horizons_eff],
        'start_dates': START_DATES,
        'config': body_base,
        'runs': runs,
        'skipped': skipped,
        'summary_by_horizon': summary_by_horizon,
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n→ JSON: {out_path}")


if __name__ == '__main__':
    main()
