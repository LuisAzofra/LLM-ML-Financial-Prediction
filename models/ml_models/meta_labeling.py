"""López de Prado meta-labeling primitives (Advances in Financial Machine Learning).

Pure, leak-free building blocks for meta-labeling a primary trading signal:

- ``daily_vol``          trailing EWM volatility of returns (Ch.3).
- ``triple_barrier_labels``  profit-take / stop-loss / vertical-barrier labels (Ch.3).
- ``PurgedKFold``        cross-validation with purging + embargo (Ch.7).
- ``get_avg_uniqueness`` average-uniqueness sample weights (Ch.4).
- ``frac_diff_ffd``      fixed-width-window fractional differentiation (Ch.5).
- ``train_meta_model``   purged-CV out-of-fold training of an XGBoost secondary model.

Leakage policy: features fed to the model must be trailing-only (the caller's
responsibility). The LABELS produced by ``triple_barrier_labels`` look FORWARD
on purpose — they are the training target, never an input feature. ``sigma``
used to set the barriers is trailing (``daily_vol``), so the barrier widths at
event time use only past information; only the touch outcome is forward-looking.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import xgboost as xgb


def daily_vol(close: pd.Series, span: int = 20) -> pd.Series:
    """Causal EWM std of simple returns, aligned to ``close`` (trailing only)."""
    returns = close.pct_change()
    return returns.ewm(span=span).std()


def triple_barrier_labels(
    close: pd.Series,
    event_idx,
    sigma: pd.Series,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    vbar_days: int = 20,
) -> pd.DataFrame:
    """Triple-barrier labels for a set of events (López de Prado, Ch.3).

    For each integer position ``i`` in ``event_idx`` the upper/lower barriers are
    ``close[i]*(1 + pt_mult*sigma[i])`` and ``close[i]*(1 - sl_mult*sigma[i])``.
    Bars ``j`` in ``(i, min(i+vbar_days, n-1)]`` are scanned forward; label is 1
    if the upper barrier is touched first, 0 if the lower is touched first, and
    ``int(close[j_end] > close[i])`` if neither is touched (sign at the vertical
    barrier). ``sigma[i]`` is trailing; only the touch scan looks forward.

    Returns a DataFrame indexed by the event's date with columns ``label`` (0/1),
    ``t1`` (touch or vertical-barrier date) and ``ret`` (close[touch]/close[i]-1).
    """
    close_vals = close.to_numpy(dtype=float)
    sigma_vals = sigma.to_numpy(dtype=float)
    n = len(close_vals)

    out_label = []
    out_t1 = []
    out_ret = []
    out_index = []

    for i in event_idx:
        i = int(i)
        s = sigma_vals[i]
        p0 = close_vals[i]
        upper = p0 * (1.0 + pt_mult * s)
        lower = p0 * (1.0 - sl_mult * s)

        j_end = min(i + vbar_days, n - 1)
        label = None
        touch_j = j_end
        for j in range(i + 1, j_end + 1):
            pj = close_vals[j]
            hit_up = pj >= upper
            hit_dn = pj <= lower
            if hit_up and not hit_dn:
                label = 1
                touch_j = j
                break
            if hit_dn and not hit_up:
                label = 0
                touch_j = j
                break
            if hit_up and hit_dn:
                label = 1
                touch_j = j
                break

        if label is None:
            touch_j = j_end
            label = int(close_vals[touch_j] > p0)

        out_label.append(int(label))
        out_t1.append(close.index[touch_j])
        out_ret.append(close_vals[touch_j] / p0 - 1.0)
        out_index.append(close.index[i])

    return pd.DataFrame(
        {"label": out_label, "t1": out_t1, "ret": out_ret},
        index=pd.Index(out_index, name=close.index.name),
    )


class PurgedKFold:
    """K-fold CV with López de Prado purging + embargo (Ch.7).

    Given ``t1`` (Series mapping event-start date -> label-end date), each
    contiguous test fold purges from train any sample whose ``[start, t1]`` span
    overlaps the test interval, then embargoes the next
    ``ceil(embargo_pct * n_samples)`` bars after the test fold.
    ``split`` yields integer-position ``(train_idx, test_idx)`` arrays.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, t1: Optional[pd.Series] = None):
        if t1 is None:
            raise ValueError("PurgedKFold.split requires t1 (event-end times)")

        n = len(t1)
        if len(X) != n:
            raise ValueError("X and t1 must have the same length")

        starts = t1.index
        ends = pd.Series(t1.to_numpy(), index=range(n))
        starts_pos = pd.Series(starts, index=range(n))

        indices = np.arange(n)
        embargo = int(np.ceil(n * self.embargo_pct))
        fold_bounds = [(b[0], b[-1] + 1) for b in np.array_split(indices, self.n_splits)]

        for start_ix, end_ix in fold_bounds:
            test_idx = indices[start_ix:end_ix]
            test_start_time = starts[start_ix]
            test_end_time = t1.iloc[start_ix:end_ix].max()

            train_mask = np.ones(n, dtype=bool)
            train_mask[start_ix:end_ix] = False

            for k in range(n):
                if not train_mask[k]:
                    continue
                k_start = starts_pos.iloc[k]
                k_end = ends.iloc[k]
                overlaps = (k_start <= test_end_time) and (k_end >= test_start_time)
                if overlaps:
                    train_mask[k] = False

            if embargo > 0 and end_ix < n:
                emb_hi = min(end_ix + embargo, n)
                train_mask[end_ix:emb_hi] = False

            train_idx = indices[train_mask]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def get_avg_uniqueness(t1: pd.Series, bar_index: pd.Index) -> pd.Series:
    """Average-uniqueness sample weights (López de Prado, Ch.4).

    Concurrency over each bar = number of labels whose ``[start, t1]`` span covers
    that bar. Each label's weight is the mean of ``1/concurrency`` over its span.
    The returned Series (aligned to ``t1.index``) is normalized to mean 1.
    """
    bar_index = pd.Index(bar_index)
    concurrency = pd.Series(0, index=bar_index, dtype=float)

    spans = []
    for start, end in t1.items():
        lo = bar_index.searchsorted(start, side="left")
        hi = bar_index.searchsorted(end, side="right")
        spans.append((lo, hi))
        concurrency.iloc[lo:hi] += 1.0

    conc_vals = concurrency.to_numpy()
    weights = np.empty(len(t1), dtype=float)
    for k, (lo, hi) in enumerate(spans):
        seg = conc_vals[lo:hi]
        seg = seg[seg > 0]
        weights[k] = float(np.mean(1.0 / seg)) if seg.size else 1.0

    w = pd.Series(weights, index=t1.index)
    mean_w = w.mean()
    if mean_w > 0:
        w = w / mean_w
    return w


def _ffd_weights(d: float, thresh: float) -> np.ndarray:
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])


def frac_diff_ffd(series: pd.Series, d: float, thresh: float = 1e-4) -> pd.Series:
    """Fixed-width-window fractional differentiation (López de Prado, Ch.5).

    Returns a Series aligned to ``series.index`` with NaN over the warmup window
    (the first ``len(weights)-1`` observations) where the full window is unavailable.
    """
    w = _ffd_weights(d, thresh)
    width = len(w) - 1

    vals = series.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)

    for i in range(width, n):
        window = vals[i - width : i + 1]
        if np.any(np.isnan(window)):
            continue
        out[i] = float(np.dot(w, window))

    return pd.Series(out, index=series.index)


def train_meta_model(
    X,
    y,
    t1: pd.Series,
    sample_weight=None,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    seed: int = 42,
) -> dict:
    """Train an XGBoost secondary classifier via PurgedKFold, return OOF metrics.

    Fits ``XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05,
    subsample=0.8, eval_metric='logloss', random_state=seed)`` on each purged fold
    and predicts the held-out fold. Returns a dict with ``oof_proba`` (array),
    ``auc`` (roc_auc over OOF vs y), ``ic`` (Spearman of OOF proba vs y),
    ``base_rate`` (mean y) and ``n``.
    """
    X_arr = X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    y_arr = y_arr.astype(int)
    n = len(y_arr)

    if sample_weight is not None:
        sw = sample_weight.to_numpy() if hasattr(sample_weight, "to_numpy") else np.asarray(sample_weight)
        sw = sw.astype(float)
    else:
        sw = None

    oof_proba = np.full(n, np.nan)
    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)

    for train_idx, test_idx in cv.split(X_arr, y_arr, t1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        y_tr = y_arr[train_idx]
        if len(np.unique(y_tr)) < 2:
            oof_proba[test_idx] = float(y_tr.mean())
            continue

        model = xgb.XGBClassifier(
            max_depth=3,
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
        fit_kwargs = {}
        if sw is not None:
            fit_kwargs["sample_weight"] = sw[train_idx]
        model.fit(X_arr[train_idx], y_tr, **fit_kwargs)
        oof_proba[test_idx] = model.predict_proba(X_arr[test_idx])[:, 1]

    mask = ~np.isnan(oof_proba)
    y_eval = y_arr[mask]
    p_eval = oof_proba[mask]

    if len(np.unique(y_eval)) >= 2 and len(y_eval) > 1:
        auc = float(roc_auc_score(y_eval, p_eval))
    else:
        auc = float("nan")

    if len(y_eval) > 2 and np.std(p_eval) > 0 and np.std(y_eval) > 0:
        ic = float(spearmanr(p_eval, y_eval).statistic)
    else:
        ic = float("nan")

    return {
        "oof_proba": oof_proba,
        "auc": auc,
        "ic": ic,
        "base_rate": float(y_arr.mean()),
        "n": int(n),
    }
