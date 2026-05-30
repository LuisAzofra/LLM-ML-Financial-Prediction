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

import math
import statistics
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as _stats

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


def aggressive_gate(
    rows_baseline: List[Tuple[int, str, str, Dict[str, float]]],
    rows_new: List[Tuple[int, str, str, Dict[str, float]]],
    n_resamples: int = 1000,
) -> Dict[str, Any]:
    """
    Tier 4.1 — Gate "agresivo" para sweeps de aflojamiento del bot.

    ACEPTA si:
        Δ_Sharpe IC95% lower bound > 0
        Y (Δ_avg_return IC95% lower bound > +5pp  O  Δ_MaxDD ≤ +2pp)
    RECHAZO duro si:
        Δ_avg_return IC95% upper bound < -2pp  O  Δ_MaxDD < -10pp (empeora >10pp)

    Filosofía: tolera más MaxDD si compensa con upside real (significativo).
    Acordado con el usuario para esta tanda — el bot es demasiado conservador.
    """
    sh = delta_ci(rows_baseline, rows_new, 'sharpe', n_resamples, 0.95, 0)
    rt = delta_ci(rows_baseline, rows_new, 'return', n_resamples, 0.95, 1)
    dd = delta_ci(rows_baseline, rows_new, 'maxdd',  n_resamples, 0.95, 2)

    sharpe_positive = sh is not None and sh['ci_lo'] > 0.0
    return_big_win  = rt is not None and rt['ci_lo'] > 5.0
    dd_safe         = dd is not None and dd['point'] >= -2.0
    dd_disaster     = dd is not None and dd['point'] < -10.0
    return_disaster = rt is not None and rt['ci_hi'] < -2.0

    accepted = bool(sharpe_positive and (return_big_win or dd_safe))
    hard_reject = bool(dd_disaster or return_disaster)

    return {
        'delta_sharpe': sh,
        'delta_return': rt,
        'delta_maxdd':  dd,
        'sharpe_positive':  sharpe_positive,
        'return_big_win':   return_big_win,
        'dd_safe':          dd_safe,
        'dd_disaster':      dd_disaster,
        'return_disaster':  return_disaster,
        'accepted':         accepted and not hard_reject,
        'hard_reject':      hard_reject,
        'verdict':  ('ACEPTA' if (accepted and not hard_reject)
                     else ('RECHAZA-HARD' if hard_reject
                           else 'no acepta')),
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


# ─────────────────────────────────────────────────────────────────────────────
# Rigor estadístico: skill predictivo (IC, DA) y haircut por multiple-testing
# (Probabilistic / Deflated Sharpe Ratio, Bailey & López de Prado 2014).
# ─────────────────────────────────────────────────────────────────────────────


def information_coefficient(pred: Sequence[float], realized: Sequence[float]) -> float:
    """Spearman rank correlation entre predicción y retorno realizado. 0.0 si degenerado."""
    p = np.asarray(pred, dtype=float)
    r = np.asarray(realized, dtype=float)
    if p.size < 3 or r.size != p.size:
        return 0.0
    mask = np.isfinite(p) & np.isfinite(r)
    if mask.sum() < 3:
        return 0.0
    p = p[mask]
    r = r[mask]
    if np.ptp(p) == 0.0 or np.ptp(r) == 0.0:
        return 0.0
    rho = _stats.spearmanr(p, r).correlation
    if rho is None or not np.isfinite(rho):
        return 0.0
    return float(rho)


def directional_accuracy(pred: Sequence[float], realized: Sequence[float]) -> float:
    """Fracción de aciertos de signo sobre entries con realized != 0. 0.5 si no hay entries válidas."""
    p = np.asarray(pred, dtype=float)
    r = np.asarray(realized, dtype=float)
    if p.size == 0 or r.size != p.size:
        return 0.5
    mask = np.isfinite(p) & np.isfinite(r) & (r != 0.0)
    if not mask.any():
        return 0.5
    hits = np.sign(p[mask]) == np.sign(r[mask])
    return float(np.mean(hits))


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Probabilistic Sharpe Ratio (Bailey & López de Prado).

    observed_sharpe y benchmark_sharpe son Sharpe POR PERIODO (no anualizado),
    en las mismas unidades que las n_obs observaciones de retorno. n_obs es el
    número de observaciones de retorno. Devuelve 0.0 si n_obs<2 o si el radicando
    del denominador es <=0.
    """
    if n_obs < 2:
        return 0.0
    denom_arg = 1.0 - skew * observed_sharpe + ((kurtosis - 1.0) / 4.0) * observed_sharpe ** 2
    if denom_arg <= 0.0:
        return 0.0
    z = (observed_sharpe - benchmark_sharpe) * math.sqrt(n_obs - 1) / math.sqrt(denom_arg)
    return float(_stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """
    Sharpe máximo esperado bajo la hipótesis nula tras N pruebas independientes
    (aproximación por el máximo de N gaussianas). Devuelve 0.0 si sharpe_variance<=0.
    """
    if sharpe_variance <= 0.0:
        return 0.0
    gamma_e = 0.5772156649015329
    n = max(int(n_trials), 2)
    quantile = (1.0 - gamma_e) * _stats.norm.ppf(1.0 - 1.0 / n) + gamma_e * _stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(math.sqrt(sharpe_variance) * quantile)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    sharpe_variance: float,
    skew: float,
    kurtosis: float,
    n_obs: int,
) -> float:
    """
    Deflated Sharpe Ratio: PSR contra el Sharpe máximo esperado por azar tras
    n_trials configuraciones probadas. DSR>0.95 ⇒ el Sharpe sobrevive al haircut
    por multiple-testing (no es un falso positivo de data-mining).
    """
    sr_star = expected_max_sharpe(n_trials, sharpe_variance)
    return probabilistic_sharpe_ratio(observed_sharpe, sr_star, n_obs, skew, kurtosis)


def sharpe_stats_from_rows(
    rows: List[Tuple[int, str, str, Dict[str, float]]],
) -> Tuple[float, float, float, float, int]:
    """
    (mean_sharpe, var_sharpe, skew_of_returns, kurt_of_returns, n) a partir de las
    ventanas. skew/kurt (kurtosis fisher=False, normal=3) sobre los retornos por
    ventana. Nan-safe: var=0, skew=0, kurt=3 si n<3.
    """
    if not rows:
        return 0.0, 0.0, 0.0, 3.0, 0
    ms = [m for _, _, _, m in rows]
    sharpes = np.asarray([m.get('sharpe', 0.0) for m in ms], dtype=float)
    returns = np.asarray([m.get('return', 0.0) for m in ms], dtype=float)
    n = len(ms)
    mean_sharpe = float(np.mean(sharpes)) if n else 0.0
    if n < 3:
        return mean_sharpe, 0.0, 0.0, 3.0, n
    var_sharpe = float(np.var(sharpes, ddof=1))
    if np.ptp(returns) == 0.0:
        return mean_sharpe, var_sharpe, 0.0, 3.0, n
    sk = float(_stats.skew(returns, bias=False))
    ku = float(_stats.kurtosis(returns, fisher=False, bias=False))
    if not np.isfinite(sk):
        sk = 0.0
    if not np.isfinite(ku):
        ku = 3.0
    return mean_sharpe, var_sharpe, sk, ku, n


def build_horizon_summary(
    runs: List[Dict[str, Any]],
    beat_keys: Tuple[str, ...] = ('alpha_vs_spy_pct', 'alpha_vs_qqq_pct', 'alpha_vs_6040_pct'),
) -> Dict[str, Any]:
    """
    Resumen agregado de un horizonte. Construye `rows` desde los dicts por ventana,
    llama a aggregate_with_ci y añade mean_return/median_return/pct_positive/
    avg_maxdd/worst_dd y un beat_<suffix>_pct por cada clave en beat_keys
    (% de ventanas con alpha vs benchmark > 0). {'n':0} si no hay runs.
    """
    if not runs:
        return {'n': 0}
    rows: List[Tuple[int, str, str, Dict[str, float]]] = []
    for i, r in enumerate(runs):
        rows.append((
            i,
            r.get('start_date', ''),
            r.get('end_date', ''),
            {
                'return': r.get('return_pct'),
                'bh':     r.get('bh_return_pct'),
                'alpha':  r.get('alpha_pct'),
                'maxdd':  r.get('maxdd_pct'),
                'sharpe': r.get('sharpe'),
                'sortino': r.get('sortino'),
                'pf':     r.get('profit_factor'),
                'wr':     r.get('win_rate_pct'),
                'trades': r.get('total_trades'),
            },
        ))
    agg = aggregate_with_ci(rows)
    if agg is None:
        return {'n': 0}
    summary: Dict[str, Any] = dict(agg)
    summary['mean_return'] = agg['avg_return']
    summary['median_return'] = agg['median_return']
    summary['pct_positive'] = agg['pct_pos']
    summary['avg_maxdd'] = agg['avg_maxdd']
    summary['worst_dd'] = agg['worst_dd']
    for key in beat_keys:
        suffix = key.replace('alpha_vs_', '').replace('_pct', '')
        vals = [r.get(key) for r in runs if r.get(key) is not None]
        summary[f'beat_{suffix}_pct'] = 100.0 * np.mean([v > 0 for v in vals]) if vals else 0.0
    return summary


def format_horizon_table(summaries: Dict[str, Dict[str, Any]]) -> str:
    """Render monoespaciado por horizonte (mismo estilo que format_aggregate_table)."""
    cols = [
        ('horizon',     '{:<10}', lambda h, s: h),
        ('n',           '{:>4}',  lambda h, s: f"{s.get('n', 0)}"),
        ('mean_ret±CI', '{:>24}', lambda h, s: f"{s['mean_return']:+6.2f}% [{s['return_ci'][0]:+5.1f},{s['return_ci'][1]:+5.1f}]"),
        ('median_ret',  '{:>11}', lambda h, s: f"{s['median_return']:+6.2f}%"),
        ('%pos',        '{:>6}',  lambda h, s: f"{s['pct_positive']:.0f}%"),
        ('avg_DD',      '{:>9}',  lambda h, s: f"{s['avg_maxdd']:+6.2f}%"),
        ('worst_DD',    '{:>10}', lambda h, s: f"{s['worst_dd']:+6.2f}%"),
        ('Sharpe±CI',   '{:>20}', lambda h, s: f"{s['avg_sharpe']:+5.2f} [{s['sharpe_ci'][0]:+4.2f},{s['sharpe_ci'][1]:+4.2f}]"),
        ('beat_SPY%',   '{:>10}', lambda h, s: f"{s.get('beat_spy_pct', 0.0):.0f}%"),
        ('beat_QQQ%',   '{:>10}', lambda h, s: f"{s.get('beat_qqq_pct', 0.0):.0f}%"),
        ('beat_6040%',  '{:>11}', lambda h, s: f"{s.get('beat_6040_pct', 0.0):.0f}%"),
    ]
    lines = []
    header = ''.join(fmt.format(name) for name, fmt, _ in cols)
    lines.append(header)
    lines.append('─' * len(header))
    for horizon, s in summaries.items():
        if not s or s.get('n', 0) == 0:
            continue
        row = ''.join(fmt.format(getter(horizon, s)) for _, fmt, getter in cols)
        lines.append(row)
    return '\n'.join(lines)
