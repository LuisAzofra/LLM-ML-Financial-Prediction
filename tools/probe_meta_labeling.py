"""GO/NO-GO signal probe for meta-labeling the Faber-QQQ primary.

Question: does a secondary classifier predicting P(a Faber long-eligible entry hits
its profit-take before its stop within a 20-day horizon) carry ANY real signal,
out-of-fold, beyond a shuffled-label null?

Protocol (DEV-era only, QQQ 2014-01-01 -> 2021-12-31; 2022+ and lockbox untouched):
  1. Load QQQ, compute close, sma200 (rolling 200, min_periods 100), sigma (daily_vol 20).
  2. Faber long-eligible events = days with close > sma200, subsampled every 5 bars
     to limit label overlap.
  3. Triple-barrier labels (pt=sl=1*sigma, vertical barrier 20 days).
  4. Features = FeatureEngineer().create_features(df) at the event rows; drop target_*,
     non-numeric, constant, and raw price/volume level columns; drop any row with a NaN
     feature (NO imputation -> no leakage).
  5. Average-uniqueness sample weights aligned to the surviving events.
  6. train_meta_model -> real OOF AUC / IC / base rate / n.
  7. Shuffle control: permute y for seeds 1..5, rerun with the SAME t1/weights.
  8. VERDICT: SIGNAL iff real_auc > max(shuffled_auc) AND real_auc > 0.55, else NULL.

Run: cd <repo> && PYTHONPATH=. .venv/bin/python tools/probe_meta_labeling.py
"""
import os

os.environ.setdefault("TFG_DISABLE_TF", "1")
os.environ.setdefault("TFG_LIGHTWEIGHT", "1")
os.environ.setdefault("PRICE_CACHE", "1")

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from data.data_loader import FinancialDataLoader
from models.ml_models.traditional_ml import FeatureEngineer
from models.ml_models.meta_labeling import (
    daily_vol,
    triple_barrier_labels,
    get_avg_uniqueness,
    train_meta_model,
)

DEV_START = "2014-01-01"
DEV_END = "2021-12-31"
SUBSAMPLE = 5
VBAR_DAYS = 20
PT_MULT = 1.0
SL_MULT = 1.0
N_SPLITS = 5
EMBARGO_PCT = 0.01
SHUFFLE_SEEDS = [1, 2, 3, 4, 5]

RAW_LEVEL_COLS = {"Open", "High", "Low", "Close", "Adj_Close", "Volume"}


def _feature_matrix(feat: pd.DataFrame) -> pd.DataFrame:
    drop = set()
    for c in feat.columns:
        if c.startswith("target_"):
            drop.add(c)
        elif not pd.api.types.is_numeric_dtype(feat[c]):
            drop.add(c)
        elif c in RAW_LEVEL_COLS:
            drop.add(c)
    keep = [c for c in feat.columns if c not in drop]
    X = feat[keep].copy()
    nuniq = X.nunique()
    const_cols = nuniq[nuniq <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)
    return X


def main() -> int:
    print("=" * 78)
    print("META-LABELING SIGNAL PROBE — Faber-QQQ primary (DEV era only)")
    print("=" * 78)

    df = FinancialDataLoader().download_stock_data("QQQ", DEV_START, DEV_END)
    if df.empty:
        print("ERROR: no QQQ data loaded")
        return 1
    df = df[~df.index.duplicated(keep="first")].sort_index()
    print(f"QQQ bars loaded: {len(df)}  range {df.index.min().date()} -> {df.index.max().date()}")

    close = df["Close"].astype(float)
    sma200 = close.rolling(200, min_periods=100).mean()
    sigma = daily_vol(close, 20)

    eligible = (close > sma200) & sma200.notna() & sigma.notna()
    eligible_pos = np.flatnonzero(eligible.to_numpy())
    n = len(close)
    eligible_pos = eligible_pos[eligible_pos < n - 1]
    event_pos = eligible_pos[::SUBSAMPLE]
    print(
        f"Faber long-eligible bars: {int(eligible.sum())}  "
        f"-> subsampled every {SUBSAMPLE} -> {len(event_pos)} candidate events"
    )

    labels = triple_barrier_labels(
        close, event_pos, sigma, pt_mult=PT_MULT, sl_mult=SL_MULT, vbar_days=VBAR_DAYS
    )

    feat = FeatureEngineer().create_features(df)
    X_all = _feature_matrix(feat)
    print(f"Feature columns used: {X_all.shape[1]} (target_*/non-numeric/constant/raw-level dropped)")

    event_dates = labels.index
    common = event_dates.intersection(X_all.index)
    X = X_all.loc[common]
    y = labels.loc[common, "label"]
    t1 = labels.loc[common, "t1"]

    finite_mask = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    X = X.loc[finite_mask]
    y = y.loc[X.index]
    t1 = t1.loc[X.index]
    print(f"Events after feature-NaN drop + alignment: n = {len(X)}")

    if len(X) < 50:
        print("ERROR: too few events to probe reliably")
        return 1

    weights = get_avg_uniqueness(t1, close.index)
    weights = weights.loc[X.index]

    res = train_meta_model(
        X, y, t1, sample_weight=weights, n_splits=N_SPLITS, embargo_pct=EMBARGO_PCT, seed=42
    )
    real_auc = res["auc"]
    real_ic = res["ic"]

    print("-" * 78)
    print("REAL (purged-CV out-of-fold):")
    print(f"  n_events   = {res['n']}")
    print(f"  base_rate  = {res['base_rate']:.4f}  (fraction of events that hit PT first)")
    print(f"  AUC        = {real_auc:.4f}")
    print(f"  IC (spear) = {real_ic:.4f}")

    shuffled_aucs = []
    for seed in SHUFFLE_SEEDS:
        rng = np.random.default_rng(seed)
        y_perm = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
        res_s = train_meta_model(
            X, y_perm, t1, sample_weight=weights, n_splits=N_SPLITS,
            embargo_pct=EMBARGO_PCT, seed=42,
        )
        shuffled_aucs.append(res_s["auc"])
        print(f"  shuffle seed {seed}: AUC = {res_s['auc']:.4f}")

    shuffled_aucs = np.array(shuffled_aucs, dtype=float)
    sh_mean = float(np.nanmean(shuffled_aucs))
    sh_max = float(np.nanmax(shuffled_aucs))

    print("-" * 78)
    print("SHUFFLE CONTROL (label-permutation null, same t1/weights):")
    print(f"  shuffled AUC mean = {sh_mean:.4f}")
    print(f"  shuffled AUC max  = {sh_max:.4f}")

    is_signal = (real_auc > sh_max) and (real_auc > 0.55)
    verdict = "SIGNAL" if is_signal else "NULL (no edge beyond chance)"

    print("=" * 78)
    print(f"VERDICT: {verdict}")
    print(
        f"  real_auc={real_auc:.4f}  shuffled_max={sh_max:.4f}  shuffled_mean={sh_mean:.4f}  "
        f"real_ic={real_ic:.4f}  base_rate={res['base_rate']:.4f}  n={res['n']}"
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
