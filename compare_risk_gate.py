"""
compare_risk_gate.py — Tier 1.3 A/B: BASELINE vs RISK_GATED.

Mismas N ventanas (seed=42, 2 años cada una, 2018-2024) para ambas
variantes. Aplica el harness honesto Tier 1.1 (bootstrap CI, score v2,
acceptance gate).

Ambas variantes usan el sweet-spot operativo MED:
  · trend_reverse_exit, disable_atr_stop, max_holding_days=120, target_vol=0.20
La única diferencia es `use_risk_gate=True` en el segundo.
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
    acceptance_gate, aggregate_with_ci, delta_ci, format_aggregate_table,
    regime_by_bh_quintile, render_gate, render_ranking, score_v1, score_v2,
)

API = "http://localhost:5057"
N_WINDOWS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
WINDOW_YEARS = 2
SEED = 42

EARLIEST = datetime(2018, 1, 1)
LATEST   = datetime(2024, 4, 1)
DELTA_DAYS = (LATEST - EARLIEST).days

MED_BASE = {
    'trend_reverse_exit': True,
    'disable_atr_stop':   True,
    'max_holding_days':   120,
    'target_vol':         0.20,
}

VARIANTS = [
    ('BASELINE_MED',  {**MED_BASE, 'use_risk_gate': False}),
    # Versión "tight" original: muy protectora pero corta ganadores en bull runs
    ('RG_TIGHT',      {**MED_BASE, 'use_risk_gate': True,
                       'rg_target_vol': 0.15, 'rg_kill_dd': 0.12,
                       'rg_kill_pause_days': 5,
                       'rg_daily_loss': 0.03, 'rg_cap_sym': 0.25}),
    # Versión "loose": sólo dispara en crashes serios, corta pausa
    ('RG_LOOSE',      {**MED_BASE, 'use_risk_gate': True,
                       'rg_target_vol': 0.20, 'rg_kill_dd': 0.20,
                       'rg_kill_pause_days': 2,
                       'rg_daily_loss': 0.04, 'rg_cap_sym': 0.25}),
    # Versión "guard-only": SIN kill-switch, sólo caps + daily-loss + vol-overlay
    # Para esto seteamos kill_dd=0.99 (nunca dispara) y pause_days=0
    ('RG_GUARD',      {**MED_BASE, 'use_risk_gate': True,
                       'rg_target_vol': 0.18, 'rg_kill_dd': 0.99,
                       'rg_kill_pause_days': 0,
                       'rg_daily_loss': 0.04, 'rg_cap_sym': 0.25}),
]


def random_window(rng):
    days = rng.randint(0, DELTA_DAYS)
    start = EARLIEST + timedelta(days=days)
    end = start + timedelta(days=365 * WINDOW_YEARS)
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
        "use_llm":           False,
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
    rg = payload.get('risk_gate_stats', {})
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
        'rg_kill_switch': rg.get('kill_switch', 0),
        'rg_vol_clip':    rg.get('vol_overlay_clip', 0),
        'rg_daily_freeze':rg.get('daily_loss_freeze', 0),
        'rg_cap_sym':     rg.get('cap_per_symbol', 0),
        'rg_cap_total':   rg.get('cap_total_exposure', 0),
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
                rg_str = ''
                if extras.get('use_risk_gate'):
                    rg_str = f" kill={m['rg_kill_switch']} clip={m['rg_vol_clip']} freeze={m['rg_daily_freeze']}"
                print(f"  {name:<14}: ret={m['return']:+7.2f}%  DD={m['maxdd']:+6.2f}%  PF={m['pf']:5.2f}  trades={m['trades']:3d}{rg_str}  ({m['elapsed']:.0f}s)")
            except Exception as ex:
                print(f"  {name:<14}: FAIL · {ex}")
        print()

    aggs = {name: aggregate_with_ci(results[name], n_resamples=1000, seed=42)
            for name, _ in VARIANTS}

    print()
    print('═' * 124)
    print('AGREGADOS POR VARIANTE  (CIs 95% por bootstrap)')
    print('═' * 124)
    print(format_aggregate_table(aggs, with_ci=True))

    print()
    print('═' * 124)
    print('GATES DE ACEPTACIÓN  (cada variante de risk_gate vs BASELINE_MED)')
    print('═' * 124)
    gates = {}
    for name, _ in VARIANTS:
        if name == 'BASELINE_MED':
            continue
        g = acceptance_gate(results['BASELINE_MED'], results[name],
                             sharpe_thresh=0.0, maxdd_thresh_pp=2.0,
                             return_floor_pp=-1.0, n_resamples=2000)
        print(render_gate(g, f'{name} vs BASELINE_MED'))
        print()
        gates[name] = g

    print()
    ranked_v2, render_v2 = render_ranking(aggs, score_fn=score_v2,
                                           label='score_v2 (Calmar+Sharpe+alpha+%pos−DD_disp)')
    print(render_v2)

    print()
    print('─── Desglose por régimen (quintiles de retorno B&H) ───')
    for name, _ in VARIANTS:
        regimes = regime_by_bh_quintile(results[name],
                                         n_quintiles=min(5, len(results[name])))
        if not regimes:
            continue
        print(f'  {name}:')
        for q, info in regimes.items():
            bh_lo, bh_hi = info['bh_range']
            print(f"    {q}  bh∈[{bh_lo:+.1f}%,{bh_hi:+.1f}%]  ret={info['avg_return']:+.2f}%  "
                  f"Sharpe={info['avg_sharpe']:+.2f}  MaxDD={info['avg_maxdd']:+.2f}%  (n={info['count']})")

    out = {
        'seed': SEED, 'n_windows': N_WINDOWS,
        'variants': {name: extras for name, extras in VARIANTS},
        'aggregates': aggs,
        'gates':       gates,
        'rows': {
            name: [{'window': i, 'start': s, 'end': e, **m} for i, s, e, m in rs]
            for name, rs in results.items()
        },
    }
    with open('/tmp/compare_risk_gate_result.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nDetalle JSON → /tmp/compare_risk_gate_result.json")


if __name__ == '__main__':
    main()
