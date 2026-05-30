"""GO/NO-GO probe: can a local LLM act as a useful meta-gate on Faber-QQQ longs?

Question: does a local LLM's entry-quality judgment predict the triple-barrier
outcome of a Faber long-eligible entry better than chance, out of sample, beyond
a shuffled-label null? This mirrors the ML meta-labeling probe (probe_meta_labeling.py),
which was NULL (out-of-fold AUC 0.385). The event/label setup here is IDENTICAL so
the two probes are directly comparable.

Protocol (DEV-era only, QQQ 2014-01-01 -> 2021-12-31; 2022+ and lockbox untouched):
  1. Rebuild the SAME events + labels as the ML probe (imported triple_barrier_labels,
     daily_vol). Faber long-eligible bars (close > sma200), subsampled every 5 bars.
  2. For each surviving event build a CAUSAL, NORMALIZED context from trailing
     features only (<= event date). NO calendar date and NO absolute price level are
     ever shown to the LLM, so it cannot pattern-match a specific historical moment —
     it can only judge the market STATE. Features: % above 200-day average, 20-day
     annualized vol, 1m/3m/6m returns, RSI(14), distance below the 1-year high.
  3. PRE-REGISTERED single prompt (fixed before seeing any AUC — no prompt-fishing).
  4. Parse a float in [0,1] from the LLM output (robust; unparseable -> parse-failure,
     scored 0.5). Responses cached in cache/llm_gate_probe.json keyed by prompt hash.
  5. Metrics: AUC = roc_auc_score(labels, scores); IC = Spearman(scores, labels).
     Diagnostics: distinct score count, mean/std/min/max, parse-failure count.
  6. Shuffle control: permute labels for seeds 1..5, recompute AUC; report mean/max.
  7. VERDICT: SIGNAL iff AUC > max(shuffled) AND AUC > 0.55 AND distinct_scores >= 5;
     else NULL (LLM adds no edge as a meta-gate).

Run: cd <repo> && PYTHONPATH=. .venv/bin/python tools/probe_llm_gate.py
"""
import os

os.environ.setdefault("TFG_DISABLE_TF", "1")
os.environ.setdefault("TFG_LIGHTWEIGHT", "1")
os.environ.setdefault("PRICE_CACHE", "1")

import hashlib
import json
import logging
import re
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from data.data_loader import FinancialDataLoader
from models.ml_models.meta_labeling import daily_vol, triple_barrier_labels
from utils.llm_client import LLMClient

DEV_START = "2014-01-01"
DEV_END = "2021-12-31"
SUBSAMPLE = 5
VBAR_DAYS = 20
PT_MULT = 1.0
SL_MULT = 1.0
SHUFFLE_SEEDS = [1, 2, 3, 4, 5]

CACHE_PATH = os.path.join("cache", "llm_gate_probe.json")
TRADING_DAYS = 252

SYSTEM_PROMPT = (
    "You are a quantitative trading analyst. Estimate the probability a "
    "trend-following long entry is profitable over the next ~20 trading days "
    "given the market state. Output ONLY a number between 0.00 and 1.00."
)


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r") as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(cache, fh)
    os.replace(tmp, CACHE_PATH)


def _prompt_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _build_context(close: pd.Series, sma200: pd.Series, i: int) -> str:
    p0 = float(close.iloc[i])
    sma = float(sma200.iloc[i])
    pct_above_sma = (p0 / sma - 1.0) * 100.0

    ret = close.pct_change()
    vol_20d_ann = float(ret.iloc[max(0, i - 19): i + 1].std()) * np.sqrt(TRADING_DAYS) * 100.0

    def _trailing_ret(lookback: int) -> float:
        j = i - lookback
        if j < 0:
            return float("nan")
        return (p0 / float(close.iloc[j]) - 1.0) * 100.0

    ret_1m = _trailing_ret(21)
    ret_3m = _trailing_ret(63)
    ret_6m = _trailing_ret(126)

    rsi14 = float(_rsi(close, 14).iloc[i])

    window_1y = close.iloc[max(0, i - (TRADING_DAYS - 1)): i + 1]
    high_1y = float(window_1y.max())
    dist_below_high = (p0 / high_1y - 1.0) * 100.0

    lines = [
        f"pct_above_200d_avg: {pct_above_sma:+.1f}%",
        f"vol_20d_annualized: {vol_20d_ann:.1f}%",
        f"ret_1m: {ret_1m:+.1f}%",
        f"ret_3m: {ret_3m:+.1f}%",
        f"ret_6m: {ret_6m:+.1f}%",
        f"rsi_14: {rsi14:.0f}",
        f"dist_below_1y_high: {dist_below_high:+.1f}%",
    ]
    return "\n".join(lines)


def _build_prompt(context: str) -> str:
    return (
        "Market state for a long-entry candidate:\n"
        f"{context}\n"
        "Probability this long entry is profitable over the next ~20 trading days? "
        "Output ONLY a number 0.00-1.00."
    )


def _parse_score(text: str):
    if text is None:
        return None
    m = re.search(r"[-+]?\d*\.?\d+", text)
    if not m:
        return None
    try:
        val = float(m.group(0))
    except ValueError:
        return None
    if val > 1.0 and val <= 100.0:
        val = val / 100.0
    if not (0.0 <= val <= 1.0):
        return None
    return val


def main() -> int:
    logging.disable(logging.CRITICAL)

    print("=" * 78)
    print("LLM META-GATE SIGNAL PROBE - Faber-QQQ primary (DEV era only)")
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

    date_to_pos = {ts: int(p) for ts, p in zip(close.index, range(n))}
    y = labels["label"].astype(int)
    print(f"Events (event_date, label) pairs: n = {len(y)}  base_rate = {float(y.mean()):.4f}")

    cache = _load_cache()
    client = LLMClient(provider="local-gguf")

    scores = np.empty(len(y), dtype=float)
    parse_failures = 0
    n_cache_hits = 0

    print("-" * 78)
    print("Querying LLM (causal normalized state; no date, no price level)...")
    for k, event_date in enumerate(y.index):
        i = date_to_pos[event_date]
        context = _build_context(close, sma200, i)
        prompt = _build_prompt(context)
        key = _prompt_key(SYSTEM_PROMPT + "\n\n" + prompt)

        if key in cache:
            text = cache[key]
            n_cache_hits += 1
        else:
            text = client.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=8,
            ).get("text", "")
            cache[key] = text
            if (k + 1) % 25 == 0:
                _save_cache(cache)

        val = _parse_score(text)
        if val is None:
            parse_failures += 1
            val = 0.5
        scores[k] = val

    _save_cache(cache)
    print(f"LLM calls complete: {len(y)} events  (cache hits: {n_cache_hits}, "
          f"fresh: {len(y) - n_cache_hits})")

    y_arr = y.to_numpy().astype(int)
    distinct = int(np.unique(scores).size)

    if len(np.unique(y_arr)) >= 2 and np.std(scores) > 0:
        real_auc = float(roc_auc_score(y_arr, scores))
    else:
        real_auc = float("nan")

    if np.std(scores) > 0 and np.std(y_arr) > 0:
        real_ic = float(spearmanr(scores, y_arr).statistic)
    else:
        real_ic = float("nan")

    print("-" * 78)
    print("LLM SCORE DIAGNOSTICS:")
    print(f"  n_events        = {len(y)}")
    print(f"  base_rate       = {float(y_arr.mean()):.4f}  (fraction of events that hit PT first)")
    print(f"  parse_failures  = {parse_failures}  (unparseable -> scored 0.5)")
    print(f"  distinct_scores = {distinct}")
    print(f"  score mean      = {float(np.mean(scores)):.4f}")
    print(f"  score std       = {float(np.std(scores)):.4f}")
    print(f"  score min       = {float(np.min(scores)):.4f}")
    print(f"  score max       = {float(np.max(scores)):.4f}")
    print("-" * 78)
    print("REAL (LLM gate vs forward triple-barrier outcome):")
    print(f"  AUC        = {real_auc:.4f}")
    print(f"  IC (spear) = {real_ic:.4f}")

    shuffled_aucs = []
    for seed in SHUFFLE_SEEDS:
        rng = np.random.default_rng(seed)
        y_perm = rng.permutation(y_arr)
        if len(np.unique(y_perm)) >= 2 and np.std(scores) > 0:
            auc_s = float(roc_auc_score(y_perm, scores))
        else:
            auc_s = float("nan")
        shuffled_aucs.append(auc_s)
        print(f"  shuffle seed {seed}: AUC = {auc_s:.4f}")

    shuffled_aucs = np.array(shuffled_aucs, dtype=float)
    sh_mean = float(np.nanmean(shuffled_aucs))
    sh_max = float(np.nanmax(shuffled_aucs))

    print("-" * 78)
    print("SHUFFLE CONTROL (label-permutation null, same LLM scores):")
    print(f"  shuffled AUC mean = {sh_mean:.4f}")
    print(f"  shuffled AUC max  = {sh_max:.4f}")

    is_signal = (
        not np.isnan(real_auc)
        and (real_auc > sh_max)
        and (real_auc > 0.55)
        and (distinct >= 5)
    )
    verdict = "SIGNAL" if is_signal else "NULL (LLM adds no edge as a meta-gate)"

    print("=" * 78)
    print(f"VERDICT: {verdict}")
    print(
        f"  real_auc={real_auc:.4f}  shuffled_max={sh_max:.4f}  shuffled_mean={sh_mean:.4f}  "
        f"real_ic={real_ic:.4f}  distinct_scores={distinct}  parse_failures={parse_failures}  "
        f"base_rate={float(y_arr.mean()):.4f}  n={len(y)}"
    )
    print(f"  ML meta-label AUC=0.385 (NULL) vs LLM gate AUC={real_auc:.4f}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
