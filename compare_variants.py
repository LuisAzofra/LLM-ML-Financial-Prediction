"""
Compara 5 variantes de TREND mode sobre las MISMAS ventanas aleatorias
del benchmark anterior (semilla 42). Cada variante activa o desactiva
mecanismos operativos del bot:

  · BASELINE  (V1)  : ATR stop=2.0, max_hold=30, ADX≥22, no reverse exit, vol_t=0.15
  · REV       (V2)  : + trend_reverse_exit (salir en death cross)
  · FREE      (V3)  : V2 + sin ATR stop, max_hold=120
  · FABER     (V4)  : V3 + ADX off, target_vol=0.20
  · BIG_VOL   (V5)  : V3 + target_vol=0.25
  · ADX_OFF   (V6)  : baseline + sin ADX

Tier 1.1: usa `utils/backtest_metrics` para reportar bootstrap CIs, Calmar,
score v2 y desglose por régimen. Mantiene el formato JSON anterior para
compatibilidad con `validate_harness.py`.
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
    delta_ci,
    format_aggregate_table,
    regime_by_bh_quintile,
    render_ranking,
    score_v1,
    score_v2,
)

API = "http://localhost:5057"
N_WINDOWS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
WINDOW_YEARS = 2
SEED = 42  # mismo seed que compare_modes.py

EARLIEST = datetime(2018, 1, 1)
LATEST   = datetime(2024, 4, 1)
DELTA_DAYS = (LATEST - EARLIEST).days

VARIANTS = [
    ('BASELINE', {}),
    ('FREE',     {'trend_reverse_exit': True, 'disable_atr_stop': True, 'max_holding_days': 120}),
    ('BIG_VOL',  {'trend_reverse_exit': True, 'disable_atr_stop': True, 'max_holding_days': 120,
                  'target_vol': 0.25}),
    # SAFE = BIG_VOL pero MANTENIENDO el ATR stop → cinturón vs crashes
    ('SAFE',     {'trend_reverse_exit': True, 'max_holding_days': 120, 'target_vol': 0.25}),
    # MED = vol intermedio + reverse exit + sin stop (compromise)
    ('MED',      {'trend_reverse_exit': True, 'disable_atr_stop': True, 'max_holding_days': 120,
                  'target_vol': 0.20}),
]

def random_window(rng):
    days = rng.randint(0, DELTA_DAYS)
    start = EARLIEST + timedelta(days=days)
    end   = start + timedelta(days=365 * WINDOW_YEARS)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')

def call(start, end, extras):
    body = {
        "mode":              "trend",
        "start_date":        start,
        "end_date":          end,
        "initial_capital":   100_000,
        "allow_short":       False,
        "signal_percentile": 0.82,
        "kelly_scale":       0.22,
        **extras,
    }
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
        raise RuntimeError(f"variant {extras} {start}→{end} → {payload}")
    p = payload['performance']
    return {
        'return':  p['total_return_pct'],
        'bh':      p['buy_hold_return_pct'],
        'alpha':   p['total_return_pct'] - p['buy_hold_return_pct'],
        'maxdd':   p['max_drawdown_pct'],
        'sharpe':  p['sharpe_ratio'],
        'sortino': p['sortino_ratio'],
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
        print(f"── Ventana {i}/{N_WINDOWS}  {s} → {e} ──")
        for name, extras in VARIANTS:
            try:
                m = call(s, e, extras)
                results[name].append((i, s, e, m))
                print(f"  {name:9s}: ret={m['return']:+7.2f}%  DD={m['maxdd']:+6.2f}%  PF={m['pf']:5.2f}  trades={m['trades']:3d}  ({m['elapsed']:.0f}s)")
            except Exception as ex:
                print(f"  {name:9s}: FAIL · {ex}")
        print()

    # ── Tabla agregada con bootstrap CIs (Tier 1.1) ──
    aggs = {name: aggregate_with_ci(results[name], n_resamples=1000, seed=42)
            for name, _ in VARIANTS}

    print()
    print('═' * 124)
    print('AGREGADOS POR VARIANTE  (CIs 95% por bootstrap, n_resamples=1000)')
    print('═' * 124)
    print(format_aggregate_table(aggs, with_ci=True))

    # ── Score v1 (compat) y v2 (nuevo) ──
    print()
    ranked_v1, render_v1 = render_ranking(aggs, score_fn=score_v1, label='score_v1 (legacy)')
    print(render_v1)
    ranked_v2, render_v2 = render_ranking(aggs, score_fn=score_v2, label='score_v2 (Calmar+Sharpe+alpha+%pos−DD_disp)')
    print()
    print(render_v2)
    ranked = ranked_v2  # ganador oficial = score v2

    # ── Test de significancia top-2 (paired bootstrap) ──
    if len(ranked) >= 2:
        winner, runner = ranked[0][0], ranked[1][0]
        print()
        print(f'─── ¿La diferencia #1 ({winner}) vs #2 ({runner}) es significativa? ───')
        for metric in ('sharpe', 'return', 'maxdd'):
            d = delta_ci(results[runner], results[winner], metric_key=metric, n_resamples=2000, seed=hash(metric) & 0xFF)
            if d is None:
                continue
            sig = '✓ significativa' if (d['positive'] or d['negative']) else '— NO significativa al 95%'
            print(f"  Δ_{metric:<7} = {d['point']:+7.3f}  CI95 [{d['ci_lo']:+6.3f}, {d['ci_hi']:+6.3f}]   {sig}")

    # ── Régimen por quintiles de B&H (bull vs bear) ──
    print()
    print('─── Desglose por régimen (quintiles de retorno B&H) ───')
    for name, _ in VARIANTS:
        regimes = regime_by_bh_quintile(results[name], n_quintiles=min(5, len(results[name])))
        if not regimes:
            continue
        print(f'  {name}:')
        for q, info in regimes.items():
            bh_lo, bh_hi = info['bh_range']
            print(f"    {q}  bh∈[{bh_lo:+.1f}%,{bh_hi:+.1f}%]  ret={info['avg_return']:+.2f}%  "
                  f"Sharpe={info['avg_sharpe']:+.2f}  MaxDD={info['avg_maxdd']:+.2f}%  (n={info['count']})")

    # Guardar JSON (compatible con validate_harness.py: rows + aggregates + scores)
    scores_v2_dict = {name: sc for name, sc in ranked_v2}
    scores_v1_dict = {name: sc for name, sc in ranked_v1}
    out = {
        'seed': SEED,
        'n_windows': N_WINDOWS,
        'variants': {name: extras for name, extras in VARIANTS},
        'aggregates': aggs,
        'scores': scores_v2_dict,
        'scores_v1': scores_v1_dict,
        'winner': ranked_v2[0][0],
        'winner_v1': ranked_v1[0][0],
        'rows': {
            name: [
                {'window': i, 'start': s, 'end': e, **m}
                for i, s, e, m in rs
            ]
            for name, rs in results.items()
        },
    }
    with open('/tmp/compare_variants_result.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nDetalle JSON → /tmp/compare_variants_result.json")
    print(f"\nGANADOR (score v2): {ranked_v2[0][0]}")
    if ranked_v1[0][0] != ranked_v2[0][0]:
        print(f"  (legacy v1 daba: {ranked_v1[0][0]})")

if __name__ == '__main__':
    main()
