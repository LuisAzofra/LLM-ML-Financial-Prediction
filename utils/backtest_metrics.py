"""
backtest_metrics.py — métricas robustas y harness de validación honesta.

Centraliza:
- Bootstrap CI (Sharpe, retorno, MaxDD, alpha) sobre N ventanas
- Métricas adicionales: Calmar, Ulcer Index, MaxDD dispersion
- Score v2: Calmar·40 + Sharpe·20 + 0.4·avg_alpha + 0.1·%pos − 0.5·MaxDD_dispersion
- Comparación A/B con paired bootstrap (Δ_Sharpe, Δ_MaxDD, Δ_alpha con CI)
- Walk-forward expanding window helper
- Desglose por régimen (quintil de retorno B&H)
- Render tabular con CIs

Reusa estructura de `rows = list[(window_idx, start, end, metrics_dict)]` que
producen `compare_with_llm.py` y `compare_variants.py`. metrics_dict debe tener:
return, bh, alpha, maxdd, sharpe, sortino, pf, wr, trades.
"""
from __future__ import annotations

import statistics
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

GateResult = Dict[str, Any]


def bootstrap_ci(
    values: Sequence[float],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """(point_estimate, lower_bound, upper_bound) para stat_fn aplicado a values."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    point = float(stat_fn(arr))
    rng = np.random.default_rng(seed)
    n = len(arr)
    if n == 1:
        return point, point, point
    samples = rng.choice(arr, size=(n_resamples, n), replace=True)
    stats = np.apply_along_axis(stat_fn, 1, samples)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(stats, alpha))
    hi = float(np.quantile(stats, 1.0 - alpha))
    return point, lo, hi


def calmar(avg_return: float, worst_dd: float) -> float:
    """Calmar = avg_return / |MaxDD|. 0 si MaxDD ≈ 0."""
    if abs(worst_dd) < 1e-9:
        return 0.0
    return avg_return / abs(worst_dd)


def ulcer_index(equity_curve: Sequence[float]) -> float:
    """Ulcer Index = sqrt(mean(drawdown_pct²)). Penaliza DDs largos más que MaxDD."""
    if len(equity_curve) < 2:
        return 0.0
    eq = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd_pct = (eq - peak) / np.where(peak > 0, peak, 1.0) * 100.0
    return float(np.sqrt(np.mean(dd_pct ** 2)))


def aggregate_with_ci(
    rows: List[Tuple[int, str, str, Dict[str, float]]],
    n_resamples: int = 1000,
    seed: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Aggregación de N ventanas con bootstrap CI 95% sobre métricas clave.
    rows: lista de tuplas (window_idx, start, end, metrics_dict).
    """
    if not rows:
        return None
    ms = [m for _, _, _, m in rows]
    returns = [m['return'] for m in ms]
    sharpes = [m['sharpe'] for m in ms]
    maxdds = [m['maxdd'] for m in ms]
    alphas = [m.get('alpha', m['return'] - m.get('bh', 0.0)) for m in ms]
    valid_pf = [m['pf'] for m in ms if m.get('pf', 0) < 50]

    avg_ret, ret_lo, ret_hi = bootstrap_ci(returns, np.mean, n_resamples, 0.95, seed)
    avg_sh, sh_lo, sh_hi = bootstrap_ci(sharpes, np.mean, n_resamples, 0.95, seed + 1)
    avg_dd, dd_lo, dd_hi = bootstrap_ci(maxdds, np.mean, n_resamples, 0.95, seed + 2)
    avg_al, al_lo, al_hi = bootstrap_ci(alphas, np.mean, n_resamples, 0.95, seed + 3)

    worst_dd = min(maxdds)
    pct_pos = 100.0 * sum(1 for r in returns if r > 0) / len(returns)
    maxdd_disp = statistics.stdev(maxdds) if len(maxdds) > 1 else 0.0

    return {
        'n':                 len(ms),
        'avg_return':        avg_ret,
        'return_ci':         (ret_lo, ret_hi),
        'median_return':     statistics.median(returns),
        'avg_alpha':         avg_al,
        'alpha_ci':          (al_lo, al_hi),
        'avg_maxdd':         avg_dd,
        'maxdd_ci':          (dd_lo, dd_hi),
        'avg_sharpe':        avg_sh,
        'sharpe_ci':         (sh_lo, sh_hi),
        'avg_pf':            statistics.mean(valid_pf) if valid_pf else 0.0,
        'avg_wr':            statistics.mean(m.get('wr', 0.0) for m in ms),
        'avg_trades':        statistics.mean(m.get('trades', 0) for m in ms),
        'pct_pos':           pct_pos,
        'worst_dd':          worst_dd,
        'best':              max(returns),
        'worst':             min(returns),
        'calmar':            calmar(avg_ret, worst_dd),
        'maxdd_dispersion':  maxdd_disp,
    }


def score_v2(agg: Optional[Dict[str, Any]]) -> float:
    """
    Score v2: pesa Calmar (retorno por unidad de DD), Sharpe, alpha (clipeado),
    %pos y penaliza dispersión de MaxDD entre ventanas (consistencia).

    score = 40·Calmar + 20·Sharpe + 0.4·clip(avg_alpha,[-30,30]) + 0.1·%pos − 0.5·MaxDD_dispersion

    avg_alpha se clipea a ±30pp porque en eras estructuralmente alcistas
    (BTC 2018-2024 +1500%) cualquier estrategia activa razonable tiene alpha
    nominalmente muy negativo, lo que dominaría el score si no se clipea.
    El clip preserva el espíritu (premiar bot que bate B&H) sin destruir.
    """
    if agg is None:
        return float('-inf')
    alpha_clipped = max(-30.0, min(30.0, agg.get('avg_alpha', 0.0)))
    return (
        agg.get('calmar', 0.0) * 40.0
        + agg.get('avg_sharpe', 0.0) * 20.0
        + alpha_clipped * 0.4
        + agg.get('pct_pos', 0.0) * 0.1
        - agg.get('maxdd_dispersion', 0.0) * 0.5
    )


def score_v1(agg: Optional[Dict[str, Any]]) -> float:
    """Score viejo (compatibilidad): 0.6·avg_ret + 15·Sharpe + 0.2·%pos."""
    if agg is None:
        return float('-inf')
    return (
        agg.get('avg_return', 0.0) * 0.6
        + agg.get('avg_sharpe', 0.0) * 15.0
        + agg.get('pct_pos', 0.0) * 0.2
    )


def delta_ci(
    rows_a: List[Tuple[int, str, str, Dict[str, float]]],
    rows_b: List[Tuple[int, str, str, Dict[str, float]]],
    metric_key: str = 'sharpe',
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Paired bootstrap sobre Δ = b − a por ventana. Asume rows_a y rows_b cubren
    las MISMAS ventanas (mismo seed); se aparean por window_idx.
    """
    if not rows_a or not rows_b:
        return None
    by_idx_a = {w: m for w, _, _, m in rows_a}
    by_idx_b = {w: m for w, _, _, m in rows_b}
    common = sorted(set(by_idx_a) & set(by_idx_b))
    if not common:
        return None
    deltas = [by_idx_b[i].get(metric_key, 0.0) - by_idx_a[i].get(metric_key, 0.0) for i in common]
    point, lo, hi = bootstrap_ci(deltas, np.mean, n_resamples, ci, seed)
    return {
        'metric':   metric_key,
        'n':        len(deltas),
        'point':    point,
        'ci_lo':    lo,
        'ci_hi':    hi,
        'positive': lo > 0,
        'negative': hi < 0,
    }


def acceptance_gate(
    rows_baseline: List[Tuple[int, str, str, Dict[str, float]]],
    rows_new: List[Tuple[int, str, str, Dict[str, float]]],
    sharpe_thresh: float = 0.0,
    maxdd_thresh_pp: float = 2.0,
    return_floor_pp: float = -1.0,
    n_resamples: int = 1000,
) -> GateResult:
    """
    Gate de aceptación de un cambio.
    Acepta si:
        Δ_Sharpe IC95% > sharpe_thresh
        Δ_MaxDD ≤ +maxdd_thresh_pp (puntos porcentuales, MaxDD es negativo)
        Δ_avg_return IC95% lower bound > return_floor_pp
    """
    sh = delta_ci(rows_baseline, rows_new, 'sharpe', n_resamples, 0.95, 0)
    rt = delta_ci(rows_baseline, rows_new, 'return', n_resamples, 0.95, 1)
    dd = delta_ci(rows_baseline, rows_new, 'maxdd',  n_resamples, 0.95, 2)

    sh_ok = sh is not None and sh['ci_lo'] > sharpe_thresh
    rt_ok = rt is not None and rt['ci_lo'] > return_floor_pp
    dd_ok = dd is not None and dd['point'] >= -maxdd_thresh_pp  # Δ_MaxDD = b−a; queremos que no empeore mucho

    return {
        'delta_sharpe': sh,
        'delta_return': rt,
        'delta_maxdd':  dd,
        'sharpe_ok':    sh_ok,
        'return_ok':    rt_ok,
        'maxdd_ok':     dd_ok,
        'accepted':     bool(sh_ok and rt_ok and dd_ok),
    }


def regime_by_bh_quintile(
    rows: List[Tuple[int, str, str, Dict[str, float]]],
    n_quintiles: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Desglosa rows por quintiles de retorno B&H (proxy de régimen bull/bear).
    Devuelve {quintile_label: {'count', 'avg_return', 'avg_sharpe', 'avg_maxdd'}}.
    """
    if not rows:
        return {}
    ms = [(idx, m, m.get('bh', 0.0)) for idx, _, _, m in rows]
    ms.sort(key=lambda t: t[2])
    n = len(ms)
    out = {}
    for q in range(n_quintiles):
        lo = q * n // n_quintiles
        hi = (q + 1) * n // n_quintiles if q < n_quintiles - 1 else n
        bucket = ms[lo:hi]
        if not bucket:
            continue
        out[f'Q{q+1}_bh'] = {
            'count':      len(bucket),
            'bh_range':   (bucket[0][2], bucket[-1][2]),
            'avg_return': statistics.mean(b[1]['return'] for b in bucket),
            'avg_sharpe': statistics.mean(b[1]['sharpe'] for b in bucket),
            'avg_maxdd':  statistics.mean(b[1]['maxdd']  for b in bucket),
        }
    return out


def walk_forward_expanding(
    earliest: str = '2018-01-01',
    latest:   str = '2024-04-01',
    step_months: int = 6,
    eval_window_years: int = 2,
) -> List[Tuple[str, str]]:
    """
    Genera ventanas walk-forward expanding. La ventana de EVALUACIÓN es de
    `eval_window_years` y se desliza hacia adelante en pasos de `step_months`.
    Devuelve lista de (eval_start, eval_end). El bot internamente hace su
    propio split train/test sobre cada ventana.

    Ejemplo (default): ventanas [2018-01-01→2020-01-01], [2018-07-01→2020-07-01], …
    """
    fmt = '%Y-%m-%d'
    e0 = datetime.strptime(earliest, fmt)
    eN = datetime.strptime(latest, fmt)
    out: List[Tuple[str, str]] = []
    cur = e0
    eval_days = 365 * eval_window_years
    while cur + timedelta(days=eval_days) <= eN:
        end = cur + timedelta(days=eval_days)
        out.append((cur.strftime(fmt), end.strftime(fmt)))
        # avanzar step_months
        m = cur.month - 1 + step_months
        cur = cur.replace(year=cur.year + m // 12, month=m % 12 + 1, day=1)
    return out


def format_aggregate_table(
    aggregates: Dict[str, Optional[Dict[str, Any]]],
    with_ci: bool = True,
) -> str:
    """Render tabular de aggregates con CIs si with_ci=True."""
    if with_ci:
        cols = [
            ('variant',     '{:<13}', lambda n, a: n),
            ('avg_ret±CI',  '{:>22}', lambda n, a: f"{a['avg_return']:+6.2f}% [{a['return_ci'][0]:+5.1f},{a['return_ci'][1]:+5.1f}]"),
            ('Sharpe±CI',   '{:>20}', lambda n, a: f"{a['avg_sharpe']:+5.2f} [{a['sharpe_ci'][0]:+4.2f},{a['sharpe_ci'][1]:+4.2f}]"),
            ('Calmar',      '{:>8}',  lambda n, a: f"{a['calmar']:+5.2f}"),
            ('MaxDD',       '{:>9}',  lambda n, a: f"{a['avg_maxdd']:+6.2f}%"),
            ('worst_DD',    '{:>10}', lambda n, a: f"{a['worst_dd']:+6.2f}%"),
            ('PF',          '{:>6}',  lambda n, a: f"{a['avg_pf']:5.2f}"),
            ('%pos',        '{:>6}',  lambda n, a: f"{a['pct_pos']:.0f}%"),
            ('alpha',       '{:>9}',  lambda n, a: f"{a['avg_alpha']:+6.2f}%"),
            ('DD_disp',     '{:>8}',  lambda n, a: f"{a['maxdd_dispersion']:5.2f}"),
        ]
    else:
        cols = [
            ('variant',     '{:<13}', lambda n, a: n),
            ('avg_ret',     '{:>9}',  lambda n, a: f"{a['avg_return']:+.2f}%"),
            ('Sharpe',      '{:>7}',  lambda n, a: f"{a['avg_sharpe']:.2f}"),
            ('Calmar',      '{:>7}',  lambda n, a: f"{a['calmar']:+.2f}"),
            ('worst_DD',    '{:>10}', lambda n, a: f"{a['worst_dd']:+.2f}%"),
            ('PF',          '{:>6}',  lambda n, a: f"{a['avg_pf']:.2f}"),
            ('%pos',        '{:>6}',  lambda n, a: f"{a['pct_pos']:.0f}%"),
            ('alpha',       '{:>9}',  lambda n, a: f"{a['avg_alpha']:+.2f}%"),
        ]

    lines = []
    header = ''.join(fmt.format(name) for name, fmt, _ in cols)
    lines.append(header)
    lines.append('─' * len(header))
    for name, agg in aggregates.items():
        if agg is None:
            continue
        row = ''.join(fmt.format(getter(name, agg)) for _, fmt, getter in cols)
        lines.append(row)
    return '\n'.join(lines)


def render_ranking(
    aggregates: Dict[str, Optional[Dict[str, Any]]],
    score_fn: Callable[[Optional[Dict[str, Any]]], float] = score_v2,
    label: str = 'score_v2',
) -> Tuple[List[Tuple[str, float]], str]:
    """Devuelve (ranked_list, render_string) ordenado por score descendente."""
    scored = [(name, score_fn(agg)) for name, agg in aggregates.items() if agg is not None]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    lines = [f'═══ Ranking por {label} ═══']
    for rank, (name, sc) in enumerate(scored, 1):
        marker = '🏆' if rank == 1 else '  '
        lines.append(f'  {marker} #{rank}  {name:<13}  {label}={sc:+8.2f}')
    return scored, '\n'.join(lines)


def render_gate(gate: GateResult, name: str = 'NEW vs BASELINE') -> str:
    """Render del veredicto de acceptance_gate."""
    def fmt_d(d):
        if d is None:
            return 'N/A'
        return f"{d['point']:+.3f} [CI95: {d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"

    lines = [f'═══ Acceptance gate · {name} ═══']
    lines.append(f"  Δ_Sharpe       : {fmt_d(gate['delta_sharpe'])}   {'✓' if gate['sharpe_ok'] else '✗'}")
    lines.append(f"  Δ_avg_return   : {fmt_d(gate['delta_return'])}   {'✓' if gate['return_ok'] else '✗'}")
    lines.append(f"  Δ_MaxDD        : {fmt_d(gate['delta_maxdd'])}   {'✓' if gate['maxdd_ok'] else '✗'}")
    verdict = 'ACEPTADO ✓' if gate['accepted'] else 'RECHAZADO ✗'
    lines.append(f"  Veredicto      : {verdict}")
    return '\n'.join(lines)
