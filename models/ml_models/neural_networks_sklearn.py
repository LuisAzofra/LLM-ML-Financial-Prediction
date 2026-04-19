"""
Redes Neuronales con scikit-learn: MLP y RBF
=============================================
Añadidos a peticion del revisor del TFG para complementar el ensemble con
modelos de redes neuronales que no requieran TensorFlow.

  - MLPPredictor  : Red neuronal multicapa (Multi-Layer Perceptron) con sklearn
  - RBFPredictor  : SVR con kernel RBF — equivalente funcional a una red RBF
  - NeuralEnsemble: Mini-ensemble MLP+RBF con meta-learner Ridge
"""
import numpy as np
import logging
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MLP
# ─────────────────────────────────────────────────────────────────────────────

class MLPPredictor:
    """
    Red neuronal MLP para prediccion de retornos financieros.

    Arquitectura: Input → 128 → 64 → 32 → Output (1 neurona, retorno predicho)
    Activacion: ReLU interna, lineal de salida
    Regularizacion: L2 (alpha) + early stopping + dropout implicito via max_iter

    La normalizacion de features es obligatoria; esta clase la incluye internamente.
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple = (128, 64, 32),
        alpha: float = 0.001,
        learning_rate_init: float = 0.001,
        max_iter: int = 500,
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=alpha,
            learning_rate='adaptive',
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPPredictor':
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("MLPPredictor: modelo no entrenado. Llama a .fit() primero.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        preds = self.predict(X)
        return {
            'rmse': float(np.sqrt(mean_squared_error(y, preds))),
            'mae':  float(mean_absolute_error(y, preds)),
            'r2':   float(r2_score(y, preds)),
        }

    def get_params(self) -> dict:
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'alpha': self.alpha,
            'learning_rate_init': self.learning_rate_init,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
        }


# ─────────────────────────────────────────────────────────────────────────────
# RBF (SVR con kernel radial)
# ─────────────────────────────────────────────────────────────────────────────

class RBFPredictor:
    """
    Support Vector Regression con kernel RBF.

    El kernel RBF (Radial Basis Function) es equivalente funcional a una red
    neuronal RBF de una capa oculta con numero de neuronas igual a los vectores
    soporte. Es un aproximador universal no parametrico especialmente bueno para
    capturar relaciones no lineales locales — util en mercados con regimenes.

    La normalizacion de features es obligatoria; esta clase la incluye internamente.
    """

    def __init__(
        self,
        C: float = 10.0,
        epsilon: float = 0.01,
        gamma: str = 'scale',
        kernel: str = 'rbf',
    ):
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.kernel = kernel

        self.scaler = StandardScaler()
        self.model = SVR(
            kernel=kernel,
            C=C,
            epsilon=epsilon,
            gamma=gamma,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RBFPredictor':
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("RBFPredictor: modelo no entrenado. Llama a .fit() primero.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        preds = self.predict(X)
        return {
            'rmse': float(np.sqrt(mean_squared_error(y, preds))),
            'mae':  float(mean_absolute_error(y, preds)),
            'r2':   float(r2_score(y, preds)),
        }

    def get_params(self) -> dict:
        return {
            'C': self.C,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'kernel': self.kernel,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Mini-ensemble neuronal MLP + RBF
# ─────────────────────────────────────────────────────────────────────────────

class NeuralEnsemble:
    """
    Ensemble ligero MLP + RBF con meta-learner Ridge.
    Se usa en el bot de trading como modelos adicionales al RF+XGB+LGB principal.

    Implementa walk-forward CV (TimeSeriesSplit) para metricas honestas.
    """

    def __init__(self):
        self.mlp = MLPPredictor()
        self.rbf = RBFPredictor()
        self.meta = Ridge(alpha=1.0)
        self.is_fitted = False

    def fit_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 3,
    ) -> Tuple[dict, dict]:
        """
        Entrena con walk-forward CV y devuelve metricas de cada modelo.

        Returns:
            (mlp_metrics, rbf_metrics) — dicts con rmse, mae, r2
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)

        mlp_preds_oof = np.zeros(len(y))
        rbf_preds_oof = np.zeros(len(y))

        mlp_folds, rbf_folds = [], []

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            mlp_fold = MLPPredictor()
            rbf_fold = RBFPredictor()

            try:
                mlp_fold.fit(X_tr, y_tr)
                mlp_preds_oof[test_idx] = mlp_fold.predict(X_te)
                mlp_folds.append({
                    'rmse': float(np.sqrt(mean_squared_error(y_te, mlp_preds_oof[test_idx]))),
                    'mae':  float(mean_absolute_error(y_te, mlp_preds_oof[test_idx])),
                    'r2':   float(r2_score(y_te, mlp_preds_oof[test_idx])),
                })
            except Exception as e:
                logger.warning(f"MLP fold error: {e}")
                mlp_preds_oof[test_idx] = np.mean(y_tr)

            try:
                rbf_fold.fit(X_tr, y_tr)
                rbf_preds_oof[test_idx] = rbf_fold.predict(X_te)
                rbf_folds.append({
                    'rmse': float(np.sqrt(mean_squared_error(y_te, rbf_preds_oof[test_idx]))),
                    'mae':  float(mean_absolute_error(y_te, rbf_preds_oof[test_idx])),
                    'r2':   float(r2_score(y_te, rbf_preds_oof[test_idx])),
                })
            except Exception as e:
                logger.warning(f"RBF fold error: {e}")
                rbf_preds_oof[test_idx] = np.mean(y_tr)

        # Entrenar modelos finales en todos los datos
        self.mlp.fit(X, y)
        self.rbf.fit(X, y)

        # Meta-learner sobre predicciones OOF
        oof_matrix = np.column_stack([mlp_preds_oof, rbf_preds_oof])
        self.meta.fit(oof_matrix, y)
        self.is_fitted = True

        def _avg_metrics(folds):
            if not folds:
                return {'rmse': 0.0, 'mae': 0.0, 'r2': 0.0}
            return {
                'rmse': round(float(np.mean([f['rmse'] for f in folds])), 6),
                'mae':  round(float(np.mean([f['mae']  for f in folds])), 6),
                'r2':   round(float(np.mean([f['r2']   for f in folds])), 6),
            }

        return _avg_metrics(mlp_folds), _avg_metrics(rbf_folds)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediccion combinada MLP+RBF via meta-learner."""
        if not self.is_fitted:
            raise RuntimeError("NeuralEnsemble no entrenado. Llama a fit_evaluate primero.")
        mlp_p = self.mlp.predict(X)
        rbf_p = self.rbf.predict(X)
        combined = np.column_stack([mlp_p, rbf_p])
        return self.meta.predict(combined)

    def predict_individual(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve (predicciones_MLP, predicciones_RBF) por separado."""
        return self.mlp.predict(X), self.rbf.predict(X)
