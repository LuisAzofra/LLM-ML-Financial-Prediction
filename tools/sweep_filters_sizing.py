"""
Tier 4.1 — Fase 1: sweeps A/B paramétricos para aflojar conservadurismo.

Ejecuta E1-E6 sobre las 6 ventanas seed=42 de compare_with_llm.py:
  E1: signal_percentile  ∈ {0.82, 0.75, 0.70, 0.65, 0.60}   (CRITICAL — rechaza 94%)
  E2: regime_adx_min     ∈ {22, 18, 15, 10, 0}              (rechaza solo 0.39%)
  E3: disable_golden_cross ∈ {False, True}                   (rechaza 0%)
  E4: kelly_scale        ∈ {0.22, 0.30, 0.40, 0.50, 0.60}   (sizing)
  E5: max_position_pct   ∈ {0.18, 0.22, 0.26, 0.30, 0.35}   (sizing)
  E6: max_concurrent     ∈ {3, 4, 5}                          (slots)

Aplica gate Agresivo de Tier 4.1 (utils/backtest_metrics.aggressive_gate).
Genera /tmp/sweep_E<n>_result.json + tabla por experimento + resumen
de ganadores para construir variante MED_AGGR en Fase 2.

Uso:
    PORT=5057 .venv/bin/python -u api.py &
    .venv/bin/python -u tools/sweep_filters_sizing.py [--exps E1,E4,E5,E6]
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from urllib import request as urlreq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.backtest_metrics import aggregate_with_ci, aggressive_gate

API = "http://localhost:5057"
N_WINDOWS = 6
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
}

# Experiment definitions: (label, dict overriding MED_BASE)
EXPERIMENTS = {
    'E1': ('signal_percentile', [{'signal_percentile': v} for v in
                                  [0.82, 0.75, 0.70, 0.65, 0.60]]),
    'E2': ('regime_adx_min',    [{'regime_adx_min': v} for v in
                                  [22, 18, 15, 10, 0]]),
    'E3': ('disable_golden_cross', [{'disable_golden_cross': v} for v in
                                     [False, True]]),
    'E4': ('kelly_scale',       [{'kelly_scale': v} for v in
                                  [0.22, 0.30, 0.40, 0.50, 0.60]]),
    'E5': ('max_position_pct',  [{'max_position_pct': v} for v in
                                  [0.18, 0.22, 0.26, 0.30, 0.35]]),
    'E6': ('max_concurrent',    [{'max_concurrent': v} for v in
                                  [3, 4, 5]]),
}


def random_window(rng):
    days = rng.randint(0, DELTA_DAYS)
    start = EARLIEST + timedelta(days=days)
    end   = start + timedelta(days=365 * WINDOW_YEARS)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def call(start, end, extras):
    body = {**MED_BASE, **extras, 'start_date': start, 'end_date': end}
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
        raise RuntimeError(f"variant {extras} {start}→{end} → {payload.get('error', payload)}")
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


def variant_label(param_name, val):
    if isinstance(val, bool):
        return f"{param_name}={'on' if val else 'off'}"
    if isinstance(val, float):
        return f"{param_name}={val:.2f}"
    return f"{param_name}={val}"


def run_experiment(exp_id, windows, baseline_results=None):
    """
    Ejecuta un experimento (5 valores × 6 ventanas) y aplica gate Agresivo
    de cada valor vs el primero (baseline=valor default).
    Si baseline_results se pasa (dict con resultados previos del valor default),
    los reusa para no repetir backtests entre experimentos.
    """
    param_name, values_list = EXPERIMENTS[exp_id]
    print(f"\n{'═'*88}")
    print(f"  EXPERIMENTO {exp_id} — sweep {param_name} ({len(values_list)} valores × {len(windows)} ventanas)")
    print(f"{'═'*88}")

    results = {}
    for v_dict in values_list:
        v = list(v_dict.values())[0]
        label = variant_label(param_name, v)
        results[label] = []

    for i, (s, e) in enumerate(windows, 1):
        print(f"  W{i}/{len(windows)}  {s}→{e}")
        for v_dict in values_list:
            v = list(v_dict.values())[0]
            label = variant_label(param_name, v)
            try:
                m = call(s, e, v_dict)
                results[label].append((i, s, e, m))
                print(f"    {label:<28}  ret={m['return']:+7.2f}%  DD={m['maxdd']:+6.2f}%  "
                      f"PF={m['pf']:5.2f}  trades={m['trades']:3d}  ({m['elapsed']:.1f}s)")
            except Exception as ex:
                print(f"    {label:<28}  FAIL · {ex}")

    # Agregados
    aggs = {label: aggregate_with_ci(rs, n_resamples=1000, seed=42)
            for label, rs in results.items()}

    # Baseline: el primer valor de la lista (o results pasados)
    baseline_label = variant_label(param_name, list(values_list[0].values())[0])
    baseline_rows  = results[baseline_label]

    # Tabla resumen
    print(f"\n  ── Resumen {exp_id} ──")
    print(f"  {'value':<28}  {'avg_ret':>22}  {'Sharpe':>18}  {'avg_DD':>9}  {'%pos':>5}  {'PF':>5}  {'trades':>6}")
    for label, agg in aggs.items():
        if agg is None: continue
        ret_lo, ret_hi = agg['return_ci']
        sh_lo,  sh_hi  = agg['sharpe_ci']
        ret = f"{agg['avg_return']:+.1f}% [{ret_lo:+.0f},{ret_hi:+.0f}]"
        sh  = f"{agg['avg_sharpe']:+.2f} [{sh_lo:+.2f},{sh_hi:+.2f}]"
        print(f"  {label:<28}  {ret:>22}  {sh:>18}  {agg['avg_maxdd']:+8.1f}%  "
              f"{agg['pct_pos']:>4.0f}%  {agg['avg_pf']:5.1f}  {agg['avg_trades']:>5.1f}")

    # Aplicar gate vs baseline
    print(f"\n  ── Gate Agresivo vs baseline ({baseline_label}) ──")
    winners = []
    for label, rows in results.items():
        if label == baseline_label or not rows:
            continue
        gate = aggressive_gate(baseline_rows, rows, n_resamples=2000)
        ds = gate['delta_sharpe'] or {}; rt = gate['delta_return'] or {}; dd = gate['delta_maxdd'] or {}
        print(f"  {label:<28}  Δ_Sharpe={ds.get('point',0):+.3f} [{ds.get('ci_lo',0):+.2f},{ds.get('ci_hi',0):+.2f}]  "
              f"Δ_ret={rt.get('point',0):+.2f}pp  Δ_DD={dd.get('point',0):+.2f}pp  → {gate['verdict']}")
        if gate['accepted']:
            winners.append((label, rows, gate, list(values_list[1:][0].values())[0] if False else None))

    # Mejor ganador (si hay): mayor Δ_return point
    best = None
    if winners:
        winners.sort(key=lambda w: w[2]['delta_return']['point'], reverse=True)
        best = winners[0]
        print(f"\n  🏆 GANADOR de {exp_id}: {best[0]}")
    else:
        print(f"\n  — Ningún valor de {exp_id} supera el gate Agresivo. Mantener default.")

    # JSON output
    out = {
        'exp_id': exp_id,
        'param_name': param_name,
        'windows': windows,
        'baseline_label': baseline_label,
        'results': {label: [{'window': i, 'start': s, 'end': e, **m} for i, s, e, m in rs]
                    for label, rs in results.items()},
        'aggregates': aggs,
        'winner': best[0] if best else None,
    }
    out_path = f"/tmp/sweep_{exp_id}_result.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  → JSON: {out_path}")

    return {
        'exp_id': exp_id,
        'param_name': param_name,
        'baseline': baseline_label,
        'winner_label': best[0] if best else None,
        'all_labels': list(results.keys()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exps', default='E1,E4,E5,E6,E2,E3',
                        help='comma-separated exp ids to run (default: E1,E4,E5,E6,E2,E3)')
    args = parser.parse_args()
    exp_ids = [e.strip() for e in args.exps.split(',')]

    rng = random.Random(SEED)
    windows = [random_window(rng) for _ in range(N_WINDOWS)]
    print(f"Ventanas (seed={SEED}, N={N_WINDOWS}):")
    for i, (s, e) in enumerate(windows, 1):
        print(f"  W{i}  {s} → {e}")

    summaries = []
    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            print(f"⚠️  Experimento desconocido: {exp_id}")
            continue
        s = run_experiment(exp_id, windows)
        summaries.append(s)

    # Resumen final
    print(f"\n{'═'*88}")
    print(f"  RESUMEN GLOBAL — ganadores por experimento")
    print(f"{'═'*88}")
    winners_for_combo = {}
    for s in summaries:
        if s['winner_label']:
            print(f"  {s['exp_id']}  {s['param_name']:<22}  ganador: {s['winner_label']}")
            winners_for_combo[s['param_name']] = s['winner_label']
        else:
            print(f"  {s['exp_id']}  {s['param_name']:<22}  — sin ganador (mantener default)")

    print(f"\n  Para Fase 2 (combinación AGGR), usar:")
    if winners_for_combo:
        for k, v in winners_for_combo.items():
            print(f"    · {v}")
    else:
        print("    · ningún cambio supera el gate individualmente — Tier 4.1 RECHAZADO antes de Fase 2.")

    with open('/tmp/sweep_summary.json', 'w') as f:
        json.dump({
            'summaries': summaries,
            'winners_for_combo': winners_for_combo,
        }, f, indent=2)
    print(f"\n  → Resumen: /tmp/sweep_summary.json")


if __name__ == '__main__':
    main()
