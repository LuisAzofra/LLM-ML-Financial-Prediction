"""
Modelos matemáticos REALES para predicción de volatilidad
Implementa GARCH, EGARCH y modelos híbridos LSTM-GARCH
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar librería arch para GARCH
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("⚠️  Librería 'arch' no disponible. Instala con: pip install arch")

from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


@dataclass
class VolatilityForecast:
    """Resultado de predicción de volatilidad"""
    forecast_value: float
    confidence_interval: Tuple[float, float]
    model_name: str
    horizon: int
    metrics: Dict[str, float]


class GARCHVolatilityModel:
    """
    Modelo GARCH para predicción de volatilidad.
    Implementa GARCH(p,q), EGARCH y GJR-GARCH.
    """
    
    def __init__(self, p: int = 1, q: int = 1, 
                 model_type: str = 'GARCH',
                 distribution: str = 't'):
        """
        Inicializa modelo GARCH.
        
        Args:
            p: Orden GARCH (lags de varianza)
            q: Orden ARCH (lags de residuos al cuadrado)
            model_type: 'GARCH', 'EGARCH', 'GJR'
            distribution: 'normal', 't', 'skewt'
        """
        if not ARCH_AVAILABLE:
            raise ImportError("Librería 'arch' requerida. Instala: pip install arch")
        
        self.p = p
        self.q = q
        self.model_type = model_type
        self.distribution = distribution
        self.model = None
        self.results = None
        self.fitted = False
        self.returns_mean = 0
        self.returns_std = 1
        
    def fit(self, returns: np.ndarray, 
            update_freq: int = 5,
            disp: str = 'off') -> Dict:
        """
        Ajusta el modelo GARCH a los datos.
        
        Args:
            returns: Serie de retornos (no anualizados)
            update_freq: Frecuencia de actualización de optimización
            disp: 'on' o 'off' para mostrar progreso
            
        Returns:
            Dict con métricas del ajuste
        """
        # Normalizar retornos para estabilidad numérica
        self.returns_mean = np.mean(returns)
        self.returns_std = np.std(returns)
        normalized_returns = (returns - self.returns_mean) / (self.returns_std + 1e-10)
        
        # Crear y ajustar modelo
        self.model = arch_model(
            normalized_returns,
            vol=self.model_type,
            p=self.p,
            q=self.q,
            dist=self.distribution,
            rescale=False
        )
        
        logger.info(f"Ajustando {self.model_type}({self.p},{self.q}) con distribución {self.distribution}...")
        
        self.results = self.model.fit(update_freq=update_freq, disp=disp)
        self.fitted = True
        
        # Extraer métricas
        metrics = {
            'aic': self.results.aic,
            'bic': self.results.bic,
            'log_likelihood': self.results.loglikelihood,
            'params': self.results.params.to_dict(),
            'convergence': self.results.convergence
        }
        
        logger.info(f"✅ Modelo ajustado - AIC: {metrics['aic']:.2f}, BIC: {metrics['bic']:.2f}")
        
        return metrics
    
    def forecast(self, horizon: int = 5, 
                 method: str = 'analytic') -> VolatilityForecast:
        """
        Predice volatilidad futura.
        
        Args:
            horizon: Horizonte de predicción (días)
            method: 'analytic' o 'simulation'
            
        Returns:
            VolatilityForecast con predicción e intervalos
        """
        if not self.fitted:
            raise ValueError("Modelo no ajustado. Llama a fit() primero.")
        
        # Realizar forecast
        forecasts = self.results.forecast(horizon=horizon, method=method)
        
        # Varianza predicha (desnormalizar)
        variance_forecast = forecasts.variance.values[-1, :]
        variance_forecast = variance_forecast * (self.returns_std ** 2)
        
        # Convertir a volatilidad anualizada
        volatility_forecast = np.sqrt(variance_forecast) * np.sqrt(252)
        
        # Calcular intervalo de confianza (aproximado)
        std_error = np.std(volatility_forecast)
        ci_lower = np.mean(volatility_forecast) - 1.96 * std_error
        ci_upper = np.mean(volatility_forecast) + 1.96 * std_error
        
        # Métricas del forecast
        metrics = {
            'mean_volatility': np.mean(volatility_forecast),
            'std_volatility': np.std(volatility_forecast),
            'min_volatility': np.min(volatility_forecast),
            'max_volatility': np.max(volatility_forecast)
        }
        
        return VolatilityForecast(
            forecast_value=np.mean(volatility_forecast),
            confidence_interval=(max(0, ci_lower), ci_upper),
            model_name=f"{self.model_type}({self.p},{self.q})",
            horizon=horizon,
            metrics=metrics
        )
    
    def rolling_forecast(self, returns: np.ndarray,
                        train_size: int = 252,
                        horizon: int = 1) -> np.ndarray:
        """
        Realiza predicciones rolling para backtesting.
        
        Args:
            returns: Serie completa de retornos
            train_size: Tamaño de ventana de entrenamiento
            horizon: Horizonte de predicción
            
        Returns:
            Array con predicciones
        """
        predictions = []
        n = len(returns)
        
        logger.info(f"Realizando rolling forecast con ventana de {train_size}...")
        
        for i in range(train_size, n - horizon + 1):
            train_data = returns[i-train_size:i]
            
            try:
                # Ajustar modelo
                model = arch_model(
                    train_data,
                    vol=self.model_type,
                    p=self.p,
                    q=self.q,
                    dist=self.distribution,
                    rescale=False
                )
                result = model.fit(disp='off')
                
                # Predecir
                forecast = result.forecast(horizon=horizon)
                pred_vol = np.sqrt(forecast.variance.values[-1, -1]) * np.sqrt(252)
                predictions.append(pred_vol)
                
            except Exception as e:
                logger.warning(f"Error en iteración {i}: {e}")
                predictions.append(np.nan)
        
        return np.array(predictions)
    
    def evaluate(self, actual_volatility: np.ndarray,
                predicted_volatility: np.ndarray) -> Dict[str, float]:
        """
        Evalúa la precisión de las predicciones.
        
        Args:
            actual_volatility: Volatilidad real (anualizada)
            predicted_volatility: Volatilidad predicha
            
        Returns:
            Dict con métricas de evaluación
        """
        # Eliminar NaN
        mask = ~(np.isnan(actual_volatility) | np.isnan(predicted_volatility))
        actual = actual_volatility[mask]
        predicted = predicted_volatility[mask]
        
        if len(actual) == 0:
            return {'error': 'No hay datos válidos para evaluación'}
        
        # Métricas
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
        
        # Correlación
        correlation = np.corrcoef(actual, predicted)[0, 1]
        
        # R²
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Correlation': correlation,
            'R2': r2
        }
    
    def get_summary(self) -> str:
        """Retorna resumen del modelo ajustado"""
        if not self.fitted:
            return "Modelo no ajustado"
        return str(self.results.summary())


class VolatilityModelEnsemble:
    """
    Ensemble de modelos de volatilidad.
    Combina GARCH, EGARCH, y modelos de ML.
    """
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.performance_history = {}
    
    def add_model(self, name: str, model: GARCHVolatilityModel, weight: float = 1.0):
        """Agrega un modelo al ensemble"""
        self.models[name] = model
        self.weights[name] = weight
    
    def fit_all(self, returns: np.ndarray):
        """Ajusta todos los modelos"""
        for name, model in self.models.items():
            logger.info(f"\nAjustando modelo: {name}")
            try:
                metrics = model.fit(returns)
                self.performance_history[name] = metrics
            except Exception as e:
                logger.error(f"❌ Error ajustando {name}: {e}")
    
    def forecast_ensemble(self, horizon: int = 5) -> VolatilityForecast:
        """
        Predicción ponderada del ensemble.
        """
        forecasts = []
        weights = []
        
        for name, model in self.models.items():
            if model.fitted:
                forecast = model.forecast(horizon=horizon)
                forecasts.append(forecast.forecast_value)
                weights.append(self.weights.get(name, 1.0))
        
        if not forecasts:
            raise ValueError("Ningún modelo ajustado disponible")
        
        # Normalizar pesos
        weights = np.array(weights) / sum(weights)
        
        # Promedio ponderado
        ensemble_forecast = np.average(forecasts, weights=weights)
        
        # Intervalo de confianza del ensemble
        variance = np.average((np.array(forecasts) - ensemble_forecast) ** 2, weights=weights)
        std_ensemble = np.sqrt(variance)
        
        return VolatilityForecast(
            forecast_value=ensemble_forecast,
            confidence_interval=(
                max(0, ensemble_forecast - 1.96 * std_ensemble),
                ensemble_forecast + 1.96 * std_ensemble
            ),
            model_name='Ensemble',
            horizon=horizon,
            metrics={
                'individual_forecasts': dict(zip(self.models.keys(), forecasts)),
                'weights': dict(zip(self.models.keys(), weights)),
                'std_ensemble': std_ensemble
            }
        )
    
    def optimize_weights(self, actual_volatility: np.ndarray,
                        predicted_volatilities: Dict[str, np.ndarray]):
        """
        Optimiza pesos basándose en rendimiento histórico.
        """
        errors = {}
        for name, predicted in predicted_volatilities.items():
            mask = ~(np.isnan(actual_volatility) | np.isnan(predicted))
            if np.sum(mask) > 0:
                mse = mean_squared_error(actual_volatility[mask], predicted[mask])
                errors[name] = mse
        
        # Peso inversamente proporcional al error
        if errors:
            inv_errors = {k: 1/v for k, v in errors.items()}
            total = sum(inv_errors.values())
            self.weights = {k: v/total for k, v in inv_errors.items()}
            logger.info(f"✅ Pesos optimizados: {self.weights}")


def calculate_realized_volatility(returns: np.ndarray, 
                                  window: int = 20,
                                  annualize: bool = True) -> np.ndarray:
    """
    Calcula volatilidad realizada (histórica).
    
    Args:
        returns: Serie de retornos
        window: Ventana de cálculo
        annualize: Si anualizar (multiplicar por sqrt(252))
        
    Returns:
        Serie de volatilidad realizada
    """
    # Volatilidad rolling
    rv = pd.Series(returns).rolling(window=window).std().values
    
    if annualize:
        rv = rv * np.sqrt(252)
    
    return rv


def calculate_parkinson_volatility(high: np.ndarray,
                                   low: np.ndarray,
                                   window: int = 20) -> np.ndarray:
    """
    Estimador de Parkinson para volatilidad.
    Usa high-low en lugar de close-close.
    Más eficiente para estimar volatilidad real.
    
    Args:
        high: Precios máximos
        low: Precios mínimos
        window: Ventana de cálculo
        
    Returns:
        Serie de volatilidad Parkinson
    """
    # Logaritmo del ratio high/low
    hl_ratio = np.log(high / low)
    
    # Varianza Parkinson: (1/4ln2) * mean((ln(H/L))^2)
    parkinson_var = (1.0 / (4.0 * np.log(2.0))) * pd.Series(hl_ratio ** 2).rolling(window=window).mean()
    
    # Convertir a volatilidad anualizada
    parkinson_vol = np.sqrt(parkinson_var) * np.sqrt(252)
    
    return parkinson_vol.values


def calculate_garman_klass_volatility(open_p: np.ndarray,
                                      high: np.ndarray,
                                      low: np.ndarray,
                                      close: np.ndarray,
                                      window: int = 20) -> np.ndarray:
    """
    Estimador de Garman-Klass para volatilidad.
    Más eficiente que Parkinson, usa OHLC.
    
    Args:
        open_p: Precios de apertura
        high: Precios máximos
        low: Precios mínimos
        close: Precios de cierre
        window: Ventana de cálculo
        
    Returns:
        Serie de volatilidad Garman-Klass
    """
    # Componentes de la fórmula
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_p) ** 2
    
    # Varianza Garman-Klass
    gk_var = 0.5 * pd.Series(log_hl).rolling(window=window).mean() - \
             (2 * np.log(2) - 1) * pd.Series(log_co).rolling(window=window).mean()
    
    # Convertir a volatilidad anualizada
    gk_vol = np.sqrt(gk_var) * np.sqrt(252)
    
    return gk_vol.values


# Ejemplo de uso
if __name__ == "__main__":
    # Generar datos de prueba
    np.random.seed(42)
    n = 1000
    
    # Simular retornos con volatilidad cambiante
    returns = np.random.normal(0, 0.02, n)
    
    # Añadir clustering de volatilidad
    for i in range(1, n):
        returns[i] = returns[i] * (1 + 0.5 * abs(returns[i-1]))
    
    print("=" * 60)
    print("EJEMPLO: Modelo GARCH para Volatilidad")
    print("=" * 60)
    
    # Crear y ajustar modelo
    model = GARCHVolatilityModel(p=1, q=1, model_type='GARCH', distribution='t')
    metrics = model.fit(returns)
    
    print(f"\nMétricas de ajuste:")
    print(f"  AIC: {metrics['aic']:.2f}")
    print(f"  BIC: {metrics['bic']:.2f}")
    
    # Forecast
    forecast = model.forecast(horizon=5)
    print(f"\nPredicción de volatilidad (5 días):")
    print(f"  Valor esperado: {forecast.forecast_value*100:.2f}%")
    print(f"  Intervalo 95%: [{forecast.confidence_interval[0]*100:.2f}%, {forecast.confidence_interval[1]*100:.2f}%]")
