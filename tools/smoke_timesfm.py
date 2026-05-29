"""Tier 6.1 Phase 0 — TimesFM smoke test sobre BTC-USD y AAPL.

Mide si el forecast de TimesFM sobre retornos H días tiene reliability_corr
suficiente para entrar como source preferente del calibrador isotónico (Tier 2.2).

Gate Phase 0 (mismo criterio que utils/calibration.fit_calibrator_from_oof):
  reliability_corr >= 0.5 AND brier_pred < brier_baseline
  para al menos UNO de los dos activos.

Si pasa → proceder a Phase 1 (integración en _train_and_predict_ml).
Si no  → documentar como Tier 6.1 RECHAZADO en PROGRESS.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from transformers import TimesFmModelForPrediction

from utils.calibration import fit_calibrator_from_oof  # noqa: E402

HORIZON_DAYS = 5         # mismo horizonte que el resto del pipeline ML
CONTEXT_DAYS = 512       # ventana de contexto que damos a TimesFM
START = "2020-01-01"
END = "2024-12-31"
STEP = 5                 # paso del walk-forward (no solapar muestras al máximo)

SYMBOLS = [("BTC-USD", "crypto"), ("AAPL", "stock")]


def load_prices(symbol: str) -> pd.Series:
    df = yf.download(symbol, start=START, end=END, auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        df = yf.Ticker(symbol).history(start=START, end=END, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned empty for {symbol}")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.astype(float)


def make_walk_forward_samples(
    closes: pd.Series, context: int, horizon: int, step: int
) -> list[tuple[np.ndarray, float]]:
    """Devuelve lista de (past_log_returns, realized_h_day_return)."""
    log_close = np.log(closes.values)
    samples = []
    n = len(log_close)
    # Need at least (context+1) past prices + (horizon) future prices.
    for end in range(context + 1, n - horizon, step):
        past = np.diff(log_close[end - context - 1 : end])  # exactly `context` returns
        if past.size != context:
            continue
        realized = float(log_close[end + horizon - 1] - log_close[end - 1])
        if not np.isfinite(past).all() or not np.isfinite(realized):
            continue
        samples.append((past.astype(np.float32), realized))
    return samples


def forecast_batch(
    model: TimesFmModelForPrediction,
    past_batch: list[np.ndarray],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast batched (N series); devuelve (mean_h[N], p_up[N])."""
    with torch.no_grad():
        tensors = [torch.from_numpy(p).float() for p in past_batch]
        out = model(tensors)
        full = out.full_predictions.cpu().numpy()  # (N, horizon, 10)
    horizon_eff = min(horizon, full.shape[1])
    mean_h = full[:, :horizon_eff, 0].sum(axis=1)  # (N,)
    quantile_sums = full[:, :horizon_eff, 1:].sum(axis=1)  # (N, 9)
    p_up = (quantile_sums > 0.0).mean(axis=1)  # (N,)
    return mean_h, p_up


def main() -> int:
    print("=" * 78)
    print("Tier 6.1 Phase 0 — TimesFM smoke (BTC-USD + AAPL, 5d return horizon)")
    print("=" * 78)

    print("Loading TimesFM 2.0 500m ...", flush=True)
    model = TimesFmModelForPrediction.from_pretrained(
        "google/timesfm-2.0-500m-pytorch", dtype=torch.float32
    )
    model.eval()
    print("  ctx_max:", model.config.context_length, "horizon_max:", model.config.horizon_length)

    overall_pass = False
    results = []

    for symbol, _kind in SYMBOLS:
        print(f"\n--- {symbol} ---", flush=True)
        prices = load_prices(symbol)
        print(f"  rows: {len(prices)}  range: {prices.index[0].date()} -> {prices.index[-1].date()}")

        samples = make_walk_forward_samples(prices, CONTEXT_DAYS, HORIZON_DAYS, STEP)
        print(f"  walk-forward samples: {len(samples)}")
        if len(samples) < 30:
            print("  SKIP (insufficient samples)")
            continue

        n = len(samples)
        preds_score = np.zeros(n, dtype=np.float32)
        preds_p_up = np.zeros(n, dtype=np.float32)
        realized = np.array([y for _, y in samples], dtype=np.float32)

        batch_size = 16
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_past = [samples[i][0] for i in range(start, end)]
            print(f"  forecast batch {start}-{end}/{n} ...", flush=True)
            mean_h, p_up = forecast_batch(model, batch_past, HORIZON_DAYS)
            preds_score[start:end] = mean_h
            preds_p_up[start:end] = p_up

        # Pre-calcular R² puro para "mean_forecast" (regressor-like).
        ss_res = float(np.sum((realized - preds_score) ** 2))
        ss_tot = float(np.sum((realized - realized.mean()) ** 2))
        r2_mean = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # Hit rate dirección puro (signo del forecast vs realized).
        hit_rate = float(np.mean(np.sign(preds_score) == np.sign(realized)))
        print(f"  >> mean_forecast vs realized: R²={r2_mean:+.4f}  hit_rate={hit_rate:.3f}")

        # Reportar dos pistas: (a) directamente la media del forecast (regressor-like),
        # (b) p_up del quantile spread (classifier-like).
        for tag, scores in [("mean_forecast", preds_score), ("p_up_from_quantiles", preds_p_up)]:
            print(f"\n  >> source={tag}")
            cal, m = fit_calibrator_from_oof(scores.astype(float), realized.astype(float), name=f"{symbol}_{tag}")
            brier_base = m.base_rate * (1.0 - m.base_rate)
            print(
                f"     brier_pred={m.brier_score:.4f}  brier_base={brier_base:.4f}  "
                f"reliability_corr={m.reliability_correlation:.3f}  base_rate={m.base_rate:.3f}  "
                f"accepted={m.accepted}"
            )
            results.append({
                "symbol": symbol,
                "source": tag,
                "brier_pred": m.brier_score,
                "brier_base": brier_base,
                "reliability_corr": m.reliability_correlation,
                "base_rate": m.base_rate,
                "accepted": m.accepted,
                "n_samples": len(samples),
            })
            if m.accepted:
                overall_pass = True

    print("\n" + "=" * 78)
    print("RESUMEN")
    print("=" * 78)
    df = pd.DataFrame(results)
    if not df.empty:
        print(df.to_string(index=False))
    print()
    print(f"VEREDICTO Phase 0 gate (reliability_corr >= 0.5 Y Brier < baseline en al menos 1 fila): "
          f"{'PASA' if overall_pass else 'NO PASA'}")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    sys.exit(main())
