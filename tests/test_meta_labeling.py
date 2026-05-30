"""Unit tests for the López de Prado meta-labeling primitives.

Covers ``triple_barrier_labels``, ``PurgedKFold`` (purge + embargo), ``frac_diff_ffd``
(d=0 identity, random-walk stationarity) and ``get_avg_uniqueness`` (overlap weights).

Runnable two ways:
  - .venv/bin/python tests/test_meta_labeling.py        (no pytest needed)
  - .venv/bin/python -m pytest tests/test_meta_labeling.py -v
"""
import os

os.environ.setdefault("TFG_DISABLE_TF", "1")
os.environ.setdefault("TFG_LIGHTWEIGHT", "1")

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from models.ml_models.meta_labeling import (
    daily_vol,
    triple_barrier_labels,
    PurgedKFold,
    get_avg_uniqueness,
    frac_diff_ffd,
)


def _bdates(n, start="2015-01-01"):
    return pd.bdate_range(start, periods=n)


def _raw_uniqueness(t1, bar_index):
    """Un-normalized average uniqueness = normalized weights * their own mean."""
    bar_index = pd.Index(bar_index)
    concurrency = pd.Series(0.0, index=bar_index)
    spans = []
    for start, end in t1.items():
        lo = bar_index.searchsorted(start, side="left")
        hi = bar_index.searchsorted(end, side="right")
        spans.append((lo, hi))
        concurrency.iloc[lo:hi] += 1.0
    conc = concurrency.to_numpy()
    raw = []
    for lo, hi in spans:
        seg = conc[lo:hi]
        seg = seg[seg > 0]
        raw.append(float(np.mean(1.0 / seg)) if seg.size else 1.0)
    return pd.Series(raw, index=t1.index)


def test_triple_barrier_up_first_labels_1():
    prices = np.concatenate([np.linspace(100, 110, 11), np.linspace(110, 95, 16)])
    close = pd.Series(prices, index=_bdates(len(prices)))
    sigma = pd.Series(0.05, index=close.index)

    out = triple_barrier_labels(close, [0], sigma, pt_mult=1.0, sl_mult=1.0, vbar_days=20)
    assert out["label"].iloc[0] == 1, f"expected upper-touch label 1, got {out['label'].iloc[0]}"
    assert out["ret"].iloc[0] > 0, "upper touch must have positive return"
    assert out["t1"].iloc[0] > close.index[0], "t1 must be after the event"


def test_triple_barrier_down_first_labels_0():
    prices = np.concatenate([np.linspace(100, 90, 11), np.linspace(90, 105, 16)])
    close = pd.Series(prices, index=_bdates(len(prices)))
    sigma = pd.Series(0.05, index=close.index)

    out = triple_barrier_labels(close, [0], sigma, pt_mult=1.0, sl_mult=1.0, vbar_days=20)
    assert out["label"].iloc[0] == 0, f"expected lower-touch label 0, got {out['label'].iloc[0]}"
    assert out["ret"].iloc[0] < 0, "lower touch must have negative return"


def test_triple_barrier_vertical_sign():
    n = 30
    close = pd.Series(100 + 0.01 * np.arange(n), index=_bdates(n))
    sigma = pd.Series(0.5, index=close.index)

    out = triple_barrier_labels(close, [0], sigma, pt_mult=1.0, sl_mult=1.0, vbar_days=5)
    assert out["t1"].iloc[0] == close.index[5], "no barrier touched -> t1 = vertical barrier"
    assert out["label"].iloc[0] == 1, "drifting up at vertical barrier -> label 1"


def test_purgedkfold_no_train_test_overlap():
    n = 100
    idx = _bdates(n)
    t1 = pd.Series([idx[min(i + 5, n - 1)] for i in range(n)], index=idx)
    X = np.zeros((n, 2))

    cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
    for train_idx, test_idx in cv.split(X, None, t1):
        test_start = t1.index[test_idx[0]]
        test_end = t1.iloc[test_idx].max()
        for k in train_idx:
            k_start = t1.index[k]
            k_end = t1.iloc[k]
            overlaps = (k_start <= test_end) and (k_end >= test_start)
            assert not overlaps, (
                f"train sample {k} [{k_start},{k_end}] overlaps test [{test_start},{test_end}]"
            )


def test_purgedkfold_embargo_count():
    n = 100
    idx = _bdates(n)
    t1 = pd.Series(idx, index=idx)
    X = np.zeros((n, 2))

    embargo_pct = 0.05
    embargo = int(np.ceil(n * embargo_pct))

    cv = PurgedKFold(n_splits=5, embargo_pct=embargo_pct)
    folds = list(cv.split(X, None, t1))

    train0, test0 = folds[0]
    end_ix = test0[-1] + 1
    embargoed = set(range(end_ix, min(end_ix + embargo, n)))
    assert embargoed, "expected a non-empty embargo region for the first fold"
    assert embargoed.isdisjoint(set(train0.tolist())), (
        "embargoed bars after the test fold must be excluded from train"
    )
    just_after = set(range(end_ix + embargo, min(end_ix + embargo + 3, n)))
    assert just_after.issubset(set(train0.tolist())), (
        "bars beyond the embargo window should be back in train (with t1=identity)"
    )


def test_fracdiff_d0_is_identity():
    rng = np.random.default_rng(0)
    series = pd.Series(100 + np.cumsum(rng.normal(0, 1, 500)), index=_bdates(500))

    fd = frac_diff_ffd(series, d=0.0, thresh=1e-4)
    both = pd.concat([series.rename("orig"), fd.rename("fd")], axis=1).dropna()
    corr = both["orig"].corr(both["fd"])
    assert corr > 0.99, f"d=0 frac-diff should ~equal original, corr={corr:.4f}"


def test_fracdiff_makes_randomwalk_more_stationary():
    rng = np.random.default_rng(7)
    rw = pd.Series(np.cumsum(rng.normal(0, 1, 1000)), index=_bdates(1000))

    p_raw = adfuller(rw.dropna().to_numpy())[1]
    fd = frac_diff_ffd(rw, d=0.4, thresh=1e-4).dropna()
    p_fd = adfuller(fd.to_numpy())[1]

    assert p_fd < p_raw, (
        f"frac-diff should lower the ADF p-value of a random walk: raw={p_raw:.4f} fd={p_fd:.4f}"
    )


def test_avg_uniqueness_non_overlapping_is_one():
    idx = _bdates(40)
    t1 = pd.Series([idx[4], idx[14]], index=[idx[0], idx[10]])

    w = get_avg_uniqueness(t1, idx)
    assert np.allclose(w.to_numpy(), 1.0, atol=1e-9), (
        f"non-overlapping labels should each have weight ~1, got {w.to_numpy()}"
    )


def test_avg_uniqueness_overlapping_is_below_one():
    idx = _bdates(40)
    overlap_t1 = pd.Series([idx[10], idx[10]], index=[idx[0], idx[0]])
    solo_t1 = pd.Series([idx[10]], index=[idx[0]])

    assert np.allclose(get_avg_uniqueness(solo_t1, idx).to_numpy(), 1.0), (
        "a single label spans bars uniquely -> weight 1"
    )

    raw = _raw_uniqueness(overlap_t1, idx)
    assert (raw.to_numpy() < 1.0 - 1e-9).all(), (
        f"two fully-overlapping labels must each have raw uniqueness < 1, got {raw.to_numpy()}"
    )
    assert np.allclose(raw.to_numpy(), 0.5, atol=1e-9), (
        f"two identical overlapping labels -> concurrency 2 -> uniqueness 0.5, got {raw.to_numpy()}"
    )


_TESTS = [
    ("TEST 1a  triple_barrier upper-first -> label 1", test_triple_barrier_up_first_labels_1),
    ("TEST 1b  triple_barrier lower-first -> label 0", test_triple_barrier_down_first_labels_0),
    ("TEST 1c  triple_barrier vertical-barrier sign", test_triple_barrier_vertical_sign),
    ("TEST 2a  PurgedKFold no train/test overlap", test_purgedkfold_no_train_test_overlap),
    ("TEST 2b  PurgedKFold embargo removes right count", test_purgedkfold_embargo_count),
    ("TEST 3a  frac_diff d=0 is identity (corr>0.99)", test_fracdiff_d0_is_identity),
    ("TEST 3b  frac_diff lowers ADF p-value of RW", test_fracdiff_makes_randomwalk_more_stationary),
    ("TEST 4a  avg_uniqueness non-overlap ~1", test_avg_uniqueness_non_overlapping_is_one),
    ("TEST 4b  avg_uniqueness full-overlap raw<1", test_avg_uniqueness_overlapping_is_below_one),
]


def _main():
    passed = 0
    failed = 0
    for name, fn in _TESTS:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\nSummary: {passed} passed, {failed} failed, {len(_TESTS)} total")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(_main())
