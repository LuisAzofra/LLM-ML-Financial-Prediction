"""
calibration.py — Tier 2.1: calibración probabilística isotónica.

Convierte el `ensemble_score` (R² × agreement) actual en `P(retorno > 0)`
real para que el sizing Kelly sea probabilísticamente correcto.

API expuesta:
- `IsotonicCalibrator(name)`: wrapper sobre sklearn.isotonic.IsotonicRegression
- `compute_oof_preds(X, y_continuous, model_factory, n_splits=5)`:
  TimeSeriesSplit out-of-fold predictions sobre un solo modelo (XGBoost
  como proxy del ensemble — XGB y LGBM están altamente correlacionados;
  usar sólo XGB ahorra ~2× tiempo de calibración).
- `fit_calibrator_from_oof(oof_scores, y_binary, name)`: fit completo
  con Brier + reliability metric. Devuelve calibrador o None si falla.
- `combine_confidence(p_calibrated, agreement, horizon_days)`: produce
  la confianza final con las penalizaciones legacy preservadas.

Diseño defensive: si la calibración falla (Brier ≥ 0.25 = no-mejor que
random), devuelve None y el caller usa la confianza legacy. Esto respeta
el principio del proyecto: cada cambio debe DEMOSTRAR mejora.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibratorMetrics:
    """Métricas para decidir si la calibración aporta valor."""
    brier_score: float            # ≤ 0.25 = mejor que random (clase 50/50)
    log_loss: float
    n_samples: int
    n_positives: int              # cuántas y=1
    base_rate: float              # n_positives / n_samples
    reliability_correlation: float  # corr(score_quintile_avg, y_avg)
    accepted: bool                # True si pasa filtros mínimos


class IsotonicCalibrator:
    """Wrapper persistible sobre sklearn.isotonic.IsotonicRegression.

    Mapea `ensemble_score → P(positive_return)`. Monotónico no-decreciente.
    """

    def __init__(self, name: str = 'default'):
        self.name = name
        self._iso = None  # type: Optional[Any]
        self.metrics: Optional[CalibratorMetrics] = None

    def fit(self, scores: np.ndarray, y_binary: np.ndarray) -> CalibratorMetrics:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss, log_loss

        scores = np.asarray(scores, dtype=float).ravel()
        y = np.asarray(y_binary, dtype=int).ravel()
        if len(scores) != len(y) or len(scores) < 30:
            self.metrics = CalibratorMetrics(
                brier_score=1.0, log_loss=1e9,
                n_samples=int(len(scores)), n_positives=int(y.sum()),
                base_rate=float(y.mean()) if len(y) else 0.5,
                reliability_correlation=0.0,
                accepted=False,
            )
            return self.metrics

        # IsotonicRegression(out_of_bounds='clip') extrapola en bordes
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(scores, y)
        self._iso = iso

        p = iso.predict(scores)
        # Evita log(0) en log_loss
        eps = 1e-6
        p_safe = np.clip(p, eps, 1 - eps)
        try:
            ll = float(log_loss(y, p_safe))
        except Exception:
            ll = 1e9
        bs = float(brier_score_loss(y, p))

        # Reliability: dividir scores en 5 quintiles y verificar que
        # la P predicha promedio y la frecuencia real son monotónicas.
        try:
            q_idx = np.argsort(scores)
            n = len(scores)
            chunk = max(n // 5, 1)
            q_pred, q_actual = [], []
            for k in range(5):
                lo, hi = k * chunk, (k + 1) * chunk if k < 4 else n
                bucket = q_idx[lo:hi]
                if len(bucket) == 0:
                    continue
                q_pred.append(float(np.mean(p[bucket])))
                q_actual.append(float(np.mean(y[bucket])))
            corr = float(np.corrcoef(q_pred, q_actual)[0, 1]) if len(q_pred) >= 2 else 0.0
        except Exception:
            corr = 0.0

        # Aceptación: Brier ≤ 0.24 (mejor que predecir base_rate fijo) y
        # reliability_correlation ≥ 0.7. Si no, marcamos accepted=False
        # pero permitimos el calibrador (caller decide).
        base_brier = float(np.mean((float(y.mean()) - y) ** 2))
        accepted = bool(bs <= base_brier and corr >= 0.5)

        self.metrics = CalibratorMetrics(
            brier_score=bs, log_loss=ll,
            n_samples=int(len(scores)), n_positives=int(y.sum()),
            base_rate=float(y.mean()),
            reliability_correlation=corr,
            accepted=accepted,
        )
        return self.metrics

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError(f"Calibrator '{self.name}' not fitted")
        return self._iso.predict(np.asarray(scores, dtype=float).ravel())

    def save(self, path: str) -> None:
        from joblib import dump
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dump({'iso': self._iso, 'metrics': self.metrics, 'name': self.name}, path)

    @classmethod
    def load(cls, path: str) -> 'IsotonicCalibrator':
        from joblib import load
        d = load(path)
        c = cls(name=d.get('name', 'loaded'))
        c._iso = d['iso']
        c.metrics = d.get('metrics')
        return c


def compute_oof_preds(
    X: np.ndarray,
    y_continuous: np.ndarray,
    model_factory: Callable[[], Any],
    n_splits: int = 5,
    gap: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula out-of-fold predictions usando TimeSeriesSplit.

    Returns (oof_preds_continuous, mask_filled). Cada fila i recibe la
    predicción del fold donde aparece como TEST. Filas no cubiertas
    quedan NaN; el caller debe filtrar.
    """
    from sklearn.model_selection import TimeSeriesSplit

    X = np.asarray(X)
    y = np.asarray(y_continuous, dtype=float)
    if len(X) != len(y) or len(X) < 50:
        return np.full(len(X), np.nan), np.zeros(len(X), dtype=bool)

    tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    oof = np.full(len(X), np.nan, dtype=float)
    mask = np.zeros(len(X), dtype=bool)
    for tr_idx, te_idx in tss.split(X):
        try:
            mdl = model_factory()
            mdl.fit(X[tr_idx], y[tr_idx])
            oof[te_idx] = mdl.predict(X[te_idx])
            mask[te_idx] = True
        except Exception as e:
            logger.warning(f"OOF fold failed: {e}")
    return oof, mask


def fit_calibrator_from_oof(
    oof_scores: np.ndarray,
    y_continuous: np.ndarray,
    name: str = 'default',
) -> Tuple[Optional[IsotonicCalibrator], CalibratorMetrics]:
    """
    Crea un IsotonicCalibrator desde OOF predictions y target binarizado
    (y > 0). Devuelve (calibrator|None, metrics).

    Si el calibrador no es accepted (Brier no mejora baseline o reliability
    correlation < 0.5), devuelve (None, metrics) — el caller debe seguir
    con la confianza legacy.
    """
    scores = np.asarray(oof_scores, dtype=float).ravel()
    y = np.asarray(y_continuous, dtype=float).ravel()
    valid = np.isfinite(scores) & np.isfinite(y)
    if valid.sum() < 30:
        return None, CalibratorMetrics(
            brier_score=1.0, log_loss=1e9,
            n_samples=int(valid.sum()), n_positives=0,
            base_rate=0.5, reliability_correlation=0.0, accepted=False,
        )
    s = scores[valid]
    yb = (y[valid] > 0).astype(int)
    cal = IsotonicCalibrator(name=name)
    metrics = cal.fit(s, yb)
    if not metrics.accepted:
        return None, metrics
    return cal, metrics


def combine_confidence(
    p_calibrated: float,
    agreement_ratio: float,
    horizon_days: int = 5,
) -> float:
    """
    Combina la P(positive) calibrada con las penalizaciones legacy.
    Devuelve `confidence` ∈ [0.10, 0.95] listo para Kelly.

    - p_calibrated: salida del IsotonicCalibrator ∈ [0,1]
    - agreement_ratio: fracción de modelos que coinciden en signo (0-1)
    - horizon_days: si > 30, aplicamos penalización por dificultad
    """
    base = float(np.clip(p_calibrated, 0.05, 0.98))
    # Penalize disagreement (mantener pena legacy con ligero ajuste)
    if agreement_ratio < 0.67:
        base *= 0.85  # menos brutal que el 0.70 legacy ya que p_calibrada es más conservadora
    if horizon_days > 30:
        base *= max(0.75, 1.0 - (horizon_days - 30) * 0.0025)
    return float(np.clip(base, 0.10, 0.95))


def calibrator_path(symbol: str, root: str) -> str:
    """Path standardizado para persistir el calibrador de un símbolo."""
    return os.path.join(root, '_calibrators', f'{symbol.upper()}.pkl')
