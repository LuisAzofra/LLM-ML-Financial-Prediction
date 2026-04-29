"""
Tier 2.3 — A/B del bull/bear/judge debate vs single-pass LLM-gate.

Sobre las MISMAS 6 ventanas (seed=42) que compare_with_llm.py, compara:

  · MED_NOLLM             — referencia sin LLM
  · MED_LLM_GGUF          — single-pass (Qwen2.5-1.5B GGUF)
  · MED_LLM_GGUF_DEBATE   — bull/bear/judge (3× LLM calls por decisión)

Aplica el gate Tier 1.1 (Δ_Sharpe IC95% > 0, Δ_MaxDD ≤ +2pp). El debate
añade ~3× latencia LLM, así que sólo se justifica si supera el gate.
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

VARIANTS = [
    ('MED_NOLLM',            {**MED_BASE, 'use_llm': False}),
    ('MED_LLM_GGUF',         {**MED_BASE, 'use_llm': True,
                              'llm_provider': LLM_PROVIDER, 'use_debate': False}),
    ('MED_LLM_GGUF_DEBATE',  {**MED_BASE, 'use_llm': True,
                              'llm_provider': LLM_PROVIDER, 'use_debate': True}),
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
    print()

    results = {name: [] for name, _ in VARIANTS}
    for i, (s, e) in enumerate(windows, 1):
        print(f"── Ventana {i}/{N_WINDOWS}  {s} → {e} ──")
        for name, extras in VARIANTS:
            try:
                m = call(s, e, extras)
                results[name].append((i, s, e, m))
                veto_str = f" veto={m['llm_vetos']}/{m['llm_calls']}" if m['llm_calls'] else ""
                print(f"  {name:<22}: ret={m['return']:+7.2f}%  DD={m['maxdd']:+6.2f}%  PF={m['pf']:5.2f}  trades={m['trades']:3d}{veto_str}  ({m['elapsed']:.0f}s)")
            except Exception as ex:
                print(f"  {name:<22}: FAIL · {ex}")
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
    print('LLM stats por variante:')
    for name, _ in VARIANTS:
        a = aggs[name]
        if a is None or 'avg_llm_calls' not in a:
            continue
        print(f"  {name:<22}  avg_llm_calls={a['avg_llm_calls']:5.1f}  avg_llm_vetos={a['avg_llm_vetos']:5.1f}")

    print()
    ranked, render = render_ranking(aggs, score_fn=score_v2,
                                    label='score_v2 (Calmar+Sharpe+alpha+%pos−DD_disp)')
    print(render)

    print()
    print('─── Tier 1.1 GATE: ¿debate aporta valor sobre single-pass? ───')
    print('  comparación: MED_LLM_GGUF_DEBATE vs MED_LLM_GGUF')
    for metric in ('sharpe', 'return', 'maxdd'):
        d = delta_ci(results['MED_LLM_GGUF'], results['MED_LLM_GGUF_DEBATE'],
                     metric_key=metric, n_resamples=2000, seed=hash(metric) & 0xFF)
        if d is None:
            continue
        sig = '✓ significativa' if (d['positive'] or d['negative']) else '— NO significativa al 95%'
        print(f"  Δ_{metric:<7} = {d['point']:+7.3f}  CI95 [{d['ci_lo']:+6.3f}, {d['ci_hi']:+6.3f}]   {sig}")

    print()
    print('─── Referencia: ¿debate aporta sobre MED_NOLLM? ───')
    for metric in ('sharpe', 'return', 'maxdd'):
        d = delta_ci(results['MED_NOLLM'], results['MED_LLM_GGUF_DEBATE'],
                     metric_key=metric, n_resamples=2000, seed=hash(metric) & 0xFF)
        if d is None:
            continue
        sig = '✓ significativa' if (d['positive'] or d['negative']) else '— NO significativa al 95%'
        print(f"  Δ_{metric:<7} = {d['point']:+7.3f}  CI95 [{d['ci_lo']:+6.3f}, {d['ci_hi']:+6.3f}]   {sig}")

    out = {
        'seed': SEED,
        'n_windows': N_WINDOWS,
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
    with open('/tmp/compare_debate_result.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nDetalle JSON → /tmp/compare_debate_result.json")


if __name__ == '__main__':
    main()
