"""Causality / no-look-ahead tests for the Faber index_trend strategy.

These tests prove that the `index_trend` backtest handler in api.py (the
all-in-above-SMA200 Faber rule, including the new T-bill / ^IRX cash-parking
extension) makes every per-day decision using ONLY information available up to
that day. A future-price leak would let a later bar change an earlier entry,
exit, or equity point.

Strategy under test (api.py, `if is_index_trend:`):
  - sma200_full = close_full.rolling(200, min_periods=100).mean(), computed
    AFTER masking idf = idf[idf.index <= end_dt].
  - At day j: bullish = (not isnan(sma)) and px > sma; enter all-in at
    px*(1+slippage); exit when px <= sma. When flat and parking_mode in
    {rate, etf}, daily risk-free interest from ^IRX (yield, shifted 1 day) is
    accrued on cash.

Test map:
  TEST 1  end-to-end truncation invariance (the core proof), parking none+rate.
  TEST 2  signal property: rolling SMA at t is invariant to future mutation.
  TEST 3  negative control: a deliberately leaky signal IS detected to change,
          proving the property test has teeth.

OUT OF SCOPE here: the ML shuffle / label-permutation (embargo) test. Faber has
no learned label to permute, so that adversarial test does not apply to this
strategy. It will live in tests/test_shuffle.py for the meta-labeling phase,
where embargoed purged CV and label shuffling become meaningful.

Runnable two ways:
  - .venv/bin/python tests/test_causality.py        (no pytest needed)
  - .venv/bin/python -m pytest tests/test_causality.py -v
"""

import os

os.environ.setdefault('TFG_DISABLE_TF', '1')
os.environ.setdefault('TFG_LIGHTWEIGHT', '1')
os.environ.setdefault('PRICE_CACHE', '1')

import numpy as np
import pandas as pd

from api import app

client = app.test_client()


def bt(start, end, **kw):
    payload = {
        'mode': 'index_trend',
        'index_symbol': 'QQQ',
        'start_date': start,
        'end_date': end,
        'initial_capital': 100000,
        'use_llm': False,
    }
    payload.update(kw)
    resp = client.post('/api/paper/autonomous-backtest', json=payload)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.get_data(as_text=True)[:300]}"
    data = resp.get_json()
    assert data.get('status') == 'success', f"status={data.get('status')} error={data.get('error')}"
    return data


def _last_value_by_date(equity_curve):
    out = {}
    for d, v in zip(equity_curve['dates'], equity_curve['values']):
        out[d] = v
    return out


def _trade_key(t):
    return (t['entry_date'], t['exit_date'], t['entry_price'], t['exit_price'], t['return_pct'])


def _check_truncation(parking_mode):
    full = bt('2018-01-02', '2021-01-04', parking_mode=parking_mode)
    short = bt('2018-01-02', '2020-01-02', parking_mode=parking_mode)

    full_map = _last_value_by_date(full['equity_curve'])
    short_map = _last_value_by_date(short['equity_curve'])
    short_last_date = short['equity_curve']['dates'][-1]

    common = [d for d in short_map if d in full_map and d != short_last_date]
    assert len(common) > 100, f"too few overlapping dates ({len(common)}) for parking_mode={parking_mode}"

    for d in common:
        sv, fv = short_map[d], full_map[d]
        assert round(sv, 2) == round(fv, 2), (
            f"[{parking_mode}] equity diverged at {d}: short={sv} full={fv}"
        )
        assert abs(sv - fv) < 1e-6 * (abs(sv) + 1e-12), (
            f"[{parking_mode}] equity diverged at {d}: short={sv} full={fv}"
        )

    full_trades = {(t['entry_date'], t['exit_date']): _trade_key(t) for t in full['trades']}
    short_before = [t for t in short['trades'] if t['exit_date'] < short_last_date]
    assert len(short_before) >= 1, f"expected at least one completed trade for parking_mode={parking_mode}"

    for t in short_before:
        key = (t['entry_date'], t['exit_date'])
        assert key in full_trades, (
            f"[{parking_mode}] trade {key} present in short but missing from full"
        )
        assert full_trades[key] == _trade_key(t), (
            f"[{parking_mode}] trade {key} differs:\n  short={_trade_key(t)}\n  full ={full_trades[key]}"
        )

    assert len(full['trades']) >= len(short_before), (
        f"[{parking_mode}] full should have >= the early short trades"
    )
    return len(common), len(short_before), len(full['trades']), short_last_date


def test_truncation_invariance_no_parking():
    n_common, n_short, n_full, slast = _check_truncation('none')
    assert n_common > 100


def test_truncation_invariance_rate_parking():
    n_common, n_short, n_full, slast = _check_truncation('rate')
    assert n_common > 100


def _synthetic_close():
    rng = np.random.default_rng(42)
    close = pd.Series(
        100 * np.cumprod(1 + rng.normal(0, 0.01, 400)),
        index=pd.bdate_range('2015-01-01', periods=400),
    )
    return close, rng


def test_signal_property_invariant_to_future():
    close, rng = _synthetic_close()
    t = 300

    sma = close.rolling(200, min_periods=100).mean()
    bull = (close > sma)

    close_mut = close.copy()
    close_mut.iloc[t + 1:] *= 2.0
    sma_mut = close_mut.rolling(200, min_periods=100).mean()
    bull_mut = (close_mut > sma_mut)

    close_shuf = close.copy()
    close_shuf.iloc[t + 1:] = close_shuf.iloc[t + 1:].to_numpy()[rng.permutation(len(close) - (t + 1))]
    sma_shuf = close_shuf.rolling(200, min_periods=100).mean()
    bull_shuf = (close_shuf > sma_shuf)

    assert np.array_equal(sma.iloc[:t + 1].values, sma_mut.iloc[:t + 1].values, equal_nan=True), (
        "SMA up to t changed under future ×2 mutation — look-ahead in rolling mean"
    )
    assert bull.iloc[:t + 1].equals(bull_mut.iloc[:t + 1]), (
        "bull signal up to t changed under future ×2 mutation — look-ahead"
    )

    assert np.array_equal(sma.iloc[:t + 1].values, sma_shuf.iloc[:t + 1].values, equal_nan=True), (
        "SMA up to t changed under future shuffle — look-ahead in rolling mean"
    )
    assert bull.iloc[:t + 1].equals(bull_shuf.iloc[:t + 1]), (
        "bull signal up to t changed under future shuffle — look-ahead"
    )


def test_negative_control_leaky_signal_is_detected():
    close, _ = _synthetic_close()
    t = 300

    bull_leak = (close > close.shift(-1))

    close_mut = close.copy()
    close_mut.iloc[t + 1:] *= 2.0
    bull_leak_mut = (close_mut > close_mut.shift(-1))

    assert bull_leak.iloc[t] != bull_leak_mut.iloc[t], (
        "negative control FAILED: a signal reading close[t+1] did NOT change when "
        "the future was mutated — the property test would not catch a real leak"
    )


_TESTS = [
    ("TEST 1a  truncation invariance (parking_mode=none)", test_truncation_invariance_no_parking),
    ("TEST 1b  truncation invariance (parking_mode=rate)", test_truncation_invariance_rate_parking),
    ("TEST 2   signal property invariant to future bars", test_signal_property_invariant_to_future),
    ("TEST 3   negative control detects a real look-ahead", test_negative_control_leaky_signal_is_detected),
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


if __name__ == '__main__':
    import sys
    sys.exit(_main())
