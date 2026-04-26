"""
validate_harness.py — auto-validación del nuevo harness (Tier 1.1).

Lee resultados históricos de `/tmp/compare_variants_result.json` y/o
`/tmp/compare_with_llm_result.json`, recalcula agregados con bootstrap CI
95% y compara ranking por score viejo vs score v2.

Veredicto humano-validable: ¿el nuevo score penaliza correctamente las
variantes con MaxDD desproporcionado y dispersión alta?

Uso:
    .venv/bin/python validate_harness.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

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

CANDIDATE_FILES = [
    '/tmp/compare_variants_result.json',
    '/tmp/compare_with_llm_result.json',
]


def _rows_from_file(path: str) -> Dict[str, List[Tuple[int, str, str, Dict[str, Any]]]]:
    """Reconstruye `rows` por variante desde el JSON guardado por compare_*.py."""
    with open(path, 'r') as f:
        data = json.load(f)
    raw = data.get('rows') or {}
    out: Dict[str, List[Tuple[int, str, str, Dict[str, Any]]]] = {}
    for variant, items in raw.items():
        rows: List[Tuple[int, str, str, Dict[str, Any]]] = []
        for item in items:
            metrics = {k: v for k, v in item.items() if k not in ('window', 'start', 'end')}
            metrics.setdefault('alpha', metrics.get('return', 0.0) - metrics.get('bh', 0.0))
            rows.append((int(item.get('window', 0)), item.get('start', ''), item.get('end', ''), metrics))
        out[variant] = rows
    return out, data


def _process_file(path: str) -> None:
    print()
    print('═' * 100)
    print(f'  ARCHIVO: {path}')
    print('═' * 100)
    rows_by_variant, raw = _rows_from_file(path)
    if not rows_by_variant:
        print('  (sin filas — saltando)')
        return

    seed = raw.get('seed')
    n = raw.get('n_windows')
    print(f'  seed={seed}  n_windows={n}  variants={list(rows_by_variant.keys())}')

    # Aggregaciones con CI
    aggs = {name: aggregate_with_ci(rows, n_resamples=1000, seed=42)
            for name, rows in rows_by_variant.items()}

    # Tabla con CIs
    print('\n' + format_aggregate_table(aggs, with_ci=True))

    # Ranking v1 vs v2
    print()
    ranked_v1, render_v1 = render_ranking(aggs, score_fn=score_v1, label='score_v1 (viejo)')
    print(render_v1)
    ranked_v2, render_v2 = render_ranking(aggs, score_fn=score_v2, label='score_v2 (nuevo)')
    print()
    print(render_v2)

    # Comparación de ranking
    print()
    print('─── Diferencias de ranking ───')
    rank_v1 = {n: i for i, (n, _) in enumerate(ranked_v1, 1)}
    rank_v2 = {n: i for i, (n, _) in enumerate(ranked_v2, 1)}
    for name in rows_by_variant.keys():
        r1, r2 = rank_v1.get(name, '?'), rank_v2.get(name, '?')
        change = ''
        if isinstance(r1, int) and isinstance(r2, int):
            if r2 < r1: change = f' ↑ subió {r1-r2}'
            elif r2 > r1: change = f' ↓ bajó {r2-r1}'
            else:        change = ' = igual'
        print(f"  {name:<14}  v1=#{r1}  →  v2=#{r2}{change}")

    # Significancia top-2 (paired bootstrap)
    if len(ranked_v2) >= 2:
        winner = ranked_v2[0][0]
        runner = ranked_v2[1][0]
        print()
        print(f'─── ¿Es la diferencia entre #{1} ({winner}) y #{2} ({runner}) significativa? ───')
        rows_w = rows_by_variant[winner]
        rows_r = rows_by_variant[runner]
        for metric in ('sharpe', 'return', 'maxdd'):
            d = delta_ci(rows_r, rows_w, metric_key=metric, n_resamples=2000, seed=hash(metric) & 0xFF)
            if d is None:
                continue
            sig = '✓ significativa' if (d['positive'] or d['negative']) else '— NO significativa al 95%'
            print(f"  Δ_{metric:<7} = {d['point']:+7.3f}  CI95 [{d['ci_lo']:+6.3f}, {d['ci_hi']:+6.3f}]   {sig}")

    # Régimen por quintil B&H
    print()
    print('─── Desglose por régimen (quintiles de B&H) ───')
    for name, rows in rows_by_variant.items():
        regimes = regime_by_bh_quintile(rows, n_quintiles=min(5, len(rows)))
        if not regimes:
            continue
        print(f'  {name}:')
        for q, info in regimes.items():
            bh_lo, bh_hi = info['bh_range']
            print(f"    {q}  bh∈[{bh_lo:+.1f}%,{bh_hi:+.1f}%]  ret={info['avg_return']:+.2f}%  "
                  f"Sharpe={info['avg_sharpe']:+.2f}  MaxDD={info['avg_maxdd']:+.2f}%  (n={info['count']})")


def main() -> int:
    found = [p for p in CANDIDATE_FILES if os.path.exists(p)]
    if not found:
        print('ERROR: no encontré ningún archivo histórico en /tmp/.')
        print('Esperaba alguno de:')
        for p in CANDIDATE_FILES:
            print(f'  - {p}')
        print('\nEjecuta primero `compare_variants.py` o `compare_with_llm.py` y reintenta.')
        return 1

    for p in found:
        _process_file(p)
    print()
    print('═' * 100)
    print('  Validación completada. Revisa que el ranking v2 ponga el peso correcto')
    print('  sobre MaxDD/Calmar (variantes muy DD-pesadas deberían perder posiciones).')
    print('═' * 100)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
