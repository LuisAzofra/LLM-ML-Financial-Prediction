"""
Tier 3.3 — Sweep del ratio ML/LLM en el sizing híbrido.

Sobre las MISMAS 6 ventanas (seed=42) que compare_with_llm.py, itera
ml_weight ∈ {0.6, 0.7, 0.8, 0.9, 1.0} con LLM activo (Qwen 1.5B GGUF).

ml_weight=1.0 equivale al comportamiento veto-only previo (LLM no afecta
sizing, sólo veta cuando contradice fuerte).

Aplica gate Tier 1.1 contra MED_LLM_GGUF (ml_weight=0.75 default) y
contra MED_NOLLM (referencia sin LLM). Acepta el ratio con mejor Calmar
si supera el gate.
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
    render_ranking,
    score_v2,
)

API = "http://localhost:5057"
N_WINDOWS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
WINDOW_YEARS = 2
SEED = 42

EARLIEST = datetime(2018, 1, 1)
LATEST   = datetime(2024, 4, 1)
DELTA_DAYS = (LATEST - EARLIEST).days

LLM_PROVIDER = os.environ.get('TFG_LLM_PROVIDER', 'local-gguf')

MED_BASE = {'trend_reverse_exit': True, 'disable_atr_stop': True,
            'max_holding_days': 120, 'target_vol': 0.20}

# Subset razonable para tiempo: 5 ratios + control sin LLM. ml_weight=1.0
# replica veto-only (LLM no afecta sizing). 0.6 da mucho peso al LLM, útil
# para detectar si la mezcla aporta o degrada.
RATIOS = [0.6, 0.7, 0.8, 0.9, 1.0]

VARIANTS = (
    [('MED_NOLLM', {**MED_BASE, 'use_llm': False})] +
    [
        (f'MED_GGUF_w{int(w*100):02d}',
         {**MED_BASE, 'use_llm': True, 'llm_provider': LLM_PROVIDER, 'ml_weight': w})
        for w in RATIOS
    ]
)


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
    with urlreq.urlopen(req, timeout=3600) as r:
        payload = json.loads(r.read())
    elapsed = time.time() - t0
    if payload.get('status') != 'success':
        raise RuntimeError(f"variant {extras} {start}→{end} → {payload}")
    p  = payload['performance']
    ls = payload.get('llm_stats') or {}
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
        'llm_calls':  ls.get('calls', 0),
        'llm_vetos':  ls.get('vetoed', 0),
        'llm_lat':    ls.get('avg_latency', 0.0),
        'elapsed': elapsed,
    }


def main():
    rng = random.Random(SEED)
    windows = [random_window(rng) for _ in range(N_WINDOWS)]
    print(f"Ventanas (seed={SEED}, N={N_WINDOWS}):")
    for i, (s, e) in enumerate(windows, 1):
        print(f"  W{i}  {s} → {e}")
    print(f"Ratios sweep: ml_weight ∈ {RATIOS}")
    print()

    results = {name: [] for name, _ in VARIANTS}
    for i, (s, e) in enumerate(windows, 1):
        print(f"── Ventana {i}/{N_WINDOWS}  {s} → {e} ──")
        for name, extras in VARIANTS:
            try:
                m = call(s, e, extras)
                results[name].append((i, s, e, m))
                veto_str = f" veto={m['llm_vetos']}/{m['llm_calls']}" if m['llm_calls'] else ""
                print(f"  {name:<16}: ret={m['return']:+7.2f}%  DD={m['maxdd']:+6.2f}%  PF={m['pf']:5.2f}  trades={m['trades']:3d}{veto_str}  ({m['elapsed']:.0f}s)")
            except Exception as ex:
                print(f"  {name:<16}: FAIL · {ex}")
        print()

    aggs = {name: aggregate_with_ci(results[name], n_resamples=1000, seed=42)
            for name, _ in VARIANTS}
    for name, _ in VARIANTS:
        rs = results[name]
        if rs and aggs[name] is not None:
            ms = [m for _, _, _, m in rs]
            aggs[name]['avg_llm_calls'] = sum(m.get('llm_calls', 0) for m in ms) / len(ms)
            aggs[name]['avg_llm_vetos'] = sum(m.get('llm_vetos', 0) for m in ms) / len(ms)

    print()
    print('═' * 124)
    print('AGREGADOS POR VARIANTE  (CIs 95% por bootstrap, n_resamples=1000)')
    print('═' * 124)
    print(format_aggregate_table(aggs, with_ci=True))

    print()
    ranked, render = render_ranking(aggs, score_fn=score_v2,
                                    label='score_v2 (Calmar+Sharpe+alpha+%pos−DD_disp)')
    print(render)

    # Mejor ratio por Calmar y por Sharpe
    print()
    print('─── Mejor ratio por Calmar (worst_DD-aware) ───')
    only_llm = [(n, a) for n, a in aggs.items() if n.startswith('MED_GGUF_') and a]
    if only_llm:
        best_calmar = max(only_llm, key=lambda x: x[1].get('calmar', 0))
        print(f"  {best_calmar[0]}: Calmar={best_calmar[1].get('calmar', 0):+.3f}  "
              f"Sharpe={best_calmar[1].get('sharpe_point', 0):+.2f}  "
              f"avg_ret={best_calmar[1].get('return_point', 0):+.2f}%")

    # Gate Tier 1.1: cada ratio vs MED_NOLLM (referencia)
    print()
    print('─── Tier 1.1 GATE: cada ml_weight vs MED_NOLLM ───')
    for name, _ in VARIANTS:
        if not name.startswith('MED_GGUF_'):
            continue
        any_sig = False
        deltas = {}
        for metric in ('sharpe', 'return', 'maxdd'):
            d = delta_ci(results['MED_NOLLM'], results[name],
                         metric_key=metric, n_resamples=2000,
                         seed=hash((metric, name)) & 0xFF)
            if d is not None:
                deltas[metric] = d
                if d['positive'] or d['negative']:
                    any_sig = True
        if not deltas:
            continue
        ds = deltas.get('sharpe', {})
        dr = deltas.get('return', {})
        dd = deltas.get('maxdd', {})
        gate_pass = (
            ds.get('positive', False) and
            (dd.get('point', 0) <= 2.0) and
            (dr.get('ci_lo', 0) > -1.0)
        )
        verdict = '✓ ACEPTA gate' if gate_pass else '✗ no acepta gate'
        print(f"  {name}:  Δ_Sharpe={ds.get('point', 0):+.3f} [{ds.get('ci_lo', 0):+.2f},{ds.get('ci_hi', 0):+.2f}]  "
              f"Δ_ret={dr.get('point', 0):+.2f}pp  Δ_DD={dd.get('point', 0):+.2f}pp  → {verdict}")

    out = {
        'seed': SEED,
        'n_windows': N_WINDOWS,
        'ratios': RATIOS,
        'variants': {name: extras for name, extras in VARIANTS},
        'aggregates': aggs,
        'rows': {
            name: [
                {'window': i, 'start': s, 'end': e, **m}
                for i, s, e, m in rs
            ]
            for name, rs in results.items()
        },
    }
    with open('/tmp/compare_hybrid_ratios_result.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nDetalle JSON → /tmp/compare_hybrid_ratios_result.json")


if __name__ == '__main__':
    main()
