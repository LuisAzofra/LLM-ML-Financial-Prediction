"""
Modelos tradicionales de Machine Learning para predicción financiera
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Ingeniería de características para modelos ML"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def create_features(self, df: pd.DataFrame, target_col: str = 'Close', 
                       prediction_horizon: int = 1) -> pd.DataFrame:
        """
        Crea características para el modelo
        """
        df = df.copy()
        
        # Retornos
        for lag in [1, 2, 3, 5, 10]:
            df[f'return_{lag}d'] = df[target_col].pct_change(lag)
        
        # Medias móviles
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df[target_col].rolling(window).mean()
            df[f'sma_ratio_{window}'] = df[target_col] / df[f'sma_{window}']
        
        # Volatilidad
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df[target_col].pct_change().rolling(window).std()
        
        # RSI
        delta = df[target_col].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df[target_col].ewm(span=12).mean()
        ema_26 = df[target_col].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bandas de Bollinger
        sma_20 = df[target_col].rolling(20).mean()
        std_20 = df[target_col].rolling(20).std()
        df['bb_upper'] = sma_20 + 2 * std_20
        df['bb_lower'] = sma_20 - 2 * std_20
        df['bb_position'] = (df[target_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volumen
        if 'Volume' in df.columns:
            df['volume_sma'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
            # On-Balance Volume
            df['obv'] = (np.sign(df[target_col].diff()) * df['Volume']).fillna(0).cumsum()
            df['obv_sma'] = df['obv'].rolling(20).mean()
            df['obv_ratio'] = df['obv'] / (df['obv_sma'].abs() + 1e-10)

        # ── NEW INDICATORS ────────────────────────────────────────────────

        # Stochastic Oscillator %K / %D (14-period)
        if all(c in df.columns for c in ['High', 'Low']):
            low_14 = df['Low'].rolling(14).min()
            high_14 = df['High'].rolling(14).max()
            df['stoch_k'] = 100 * (df[target_col] - low_14) / (high_14 - low_14 + 1e-10)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()

            # Williams %R
            df['williams_r'] = -100 * (high_14 - df[target_col]) / (high_14 - low_14 + 1e-10)

            # Average Directional Index (ADX, 14-period)
            prev_close = df[target_col].shift(1)
            tr = pd.concat([
                df['High'] - df['Low'],
                (df['High'] - prev_close).abs(),
                (df['Low'] - prev_close).abs()
            ], axis=1).max(axis=1)
            dm_plus = np.where(
                (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                np.maximum(df['High'] - df['High'].shift(1), 0), 0
            )
            dm_minus = np.where(
                (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                np.maximum(df['Low'].shift(1) - df['Low'], 0), 0
            )
            atr14 = tr.rolling(14).mean()
            di_plus = 100 * pd.Series(dm_plus, index=df.index).rolling(14).mean() / (atr14 + 1e-10)
            di_minus = 100 * pd.Series(dm_minus, index=df.index).rolling(14).mean() / (atr14 + 1e-10)
            dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)
            df['adx'] = dx.rolling(14).mean()
            df['di_plus'] = di_plus
            df['di_minus'] = di_minus

        # Ichimoku components
        df['ich_conversion'] = (df[target_col].rolling(9).max() + df[target_col].rolling(9).min()) / 2
        df['ich_base'] = (df[target_col].rolling(26).max() + df[target_col].rolling(26).min()) / 2
        df['ich_span_a'] = ((df['ich_conversion'] + df['ich_base']) / 2).shift(26)
        df['ich_above_cloud'] = (df[target_col] > df['ich_span_a']).astype(int)

        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df[target_col].pct_change(period)
            df[f'momentum_{period}'] = df[target_col] - df[target_col].shift(period)

        # ── REGIME FEATURES (critical for identifying high-volatility environments) ──

        # Volatility regime: current vol vs 1-year historical (>1 = high vol regime)
        returns_series = df[target_col].pct_change()
        hist_vol_20 = returns_series.rolling(20).std() * np.sqrt(252)
        hist_vol_long = hist_vol_20.rolling(252, min_periods=60).mean()
        df['vol_regime'] = hist_vol_20 / (hist_vol_long + 1e-10)
        # Volatility percentile vs 1-year history (0=low, 1=high)
        df['vol_percentile'] = hist_vol_20.rolling(252, min_periods=60).rank(pct=True)

        # Trend regime: bull (>0.5) or bear (<0.5) based on price vs long-term SMA
        if len(df) > 200:
            sma_200 = df[target_col].rolling(200).mean()
            df['bull_market'] = (df[target_col] > sma_200).astype(float)
            df['sma_200_dist'] = (df[target_col] - sma_200) / (sma_200.abs() + 1e-10)
        else:
            df['bull_market'] = 0.5
            df['sma_200_dist'] = 0.0

        # Mean-reversion z-score: how far price is from its 20-day mean in std units
        sma_20_rv = df[target_col].rolling(20).mean()
        std_20_rv = df[target_col].rolling(20).std()
        df['price_zscore_20'] = (df[target_col] - sma_20_rv) / (std_20_rv + 1e-10)

        # Multi-timeframe momentum (monthly, quarterly, semi-annual)
        for months, label in [(1, '1m'), (3, '3m'), (6, '6m')]:
            trading_days = months * 21
            df[f'mom_{label}'] = df[target_col].pct_change(trading_days)

        # Trend consistency: how many of the last N days showed positive returns
        for window in [5, 10, 20]:
            df[f'trend_consistency_{window}'] = (
                returns_series.rolling(window).apply(lambda x: (x > 0).sum() / len(x), raw=True)
            )

        # Money Flow Index (MFI, 14-period) — volume-weighted RSI
        # Captura divergencia precio-volumen que OBV no detecta
        if all(c in df.columns for c in ['High', 'Low', 'Volume']):
            _tp  = (df['High'] + df['Low'] + df[target_col]) / 3
            _rmf = _tp * df['Volume']
            _tp_diff = _tp.diff()
            _pos_mf = _rmf.where(_tp_diff > 0, 0.0).rolling(14).sum()
            _neg_mf = _rmf.where(_tp_diff < 0, 0.0).rolling(14).sum()
            df['mfi'] = 100 - (100 / (1 + _pos_mf / (_neg_mf + 1e-10)))

        # Return distribution features — skewness y kurtosis de corto plazo
        # Los activos con alta skewness positiva tienden a revertir (efecto lotería)
        df['returns_skew_20']  = returns_series.rolling(20).skew()
        df['returns_kurt_20']  = returns_series.rolling(20).kurt()

        # Autocorrelación de returns (momentum vs mean-reversion local)
        df['autocorr_5'] = returns_series.rolling(20).apply(
            lambda x: float(pd.Series(x).autocorr(lag=5)) if len(x) >= 10 else 0.0, raw=False
        )

        # ── HIGH-ALPHA META-FEATURES ──────────────────────────────────────

        # RSI-2: señal de agotamiento de muy corto plazo (AQR, Connors & Alvarez)
        # Lecturas <5 = sobreventa extrema, >95 = sobrecompra extrema
        _delta2 = df[target_col].diff()
        _gain2  = _delta2.where(_delta2 > 0, 0).rolling(2).mean()
        _loss2  = (-_delta2.where(_delta2 < 0, 0)).rolling(2).mean()
        df['rsi_2'] = 100 - (100 / (1 + _gain2 / (_loss2 + 1e-10)))

        # Hurst Exponent (rolling 63 bars) — fractal dimension del precio
        # H > 0.55 → tendencia, H < 0.45 → reversión a la media, ~0.5 → random walk
        # Meta-feature crítico: condiciona todos los signals de momentum
        def _hurst_exponent(x: np.ndarray) -> float:
            """R/S analysis — estimación simple del exponente de Hurst."""
            n = len(x)
            if n < 20:
                return 0.5
            try:
                lags = [2, 4, 8, 16]
                tau = []
                for lag in lags:
                    if lag >= n:
                        continue
                    sub = x[-lag:]
                    std_sub = float(np.std(sub))
                    tau.append(std_sub if std_sub > 0 else 1e-10)
                if len(tau) < 2:
                    return 0.5
                poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
                return float(np.clip(poly[0], 0.0, 1.0))
            except Exception:
                return 0.5

        df['hurst_63'] = (
            returns_series.rolling(63, min_periods=30)
            .apply(_hurst_exponent, raw=True)
        )
        # Derived: is the current regime trending (H>0.55) or mean-reverting (H<0.45)?
        df['hurst_trending']     = (df['hurst_63'] > 0.55).astype(float)
        df['hurst_mean_rev']     = (df['hurst_63'] < 0.45).astype(float)

        # Garman-Klass volatility — usa OHLC, 5-8× más eficiente que close-to-close
        # gk_ratio > 1 → current vol elevated vs 20-day baseline (high-vol regime)
        if all(c in df.columns for c in ['High', 'Low', 'Open']):
            _log_hl = np.log(df['High'] / df['Low'])
            _log_co = np.log(df[target_col] / df['Open'])
            _gk_daily = 0.5 * _log_hl ** 2 - (2 * np.log(2) - 1) * _log_co ** 2
            df['gk_vol_5']  = np.sqrt(_gk_daily.rolling(5,  min_periods=3).mean() * 252)
            df['gk_vol_20'] = np.sqrt(_gk_daily.rolling(20, min_periods=10).mean() * 252)
            df['gk_ratio']  = df['gk_vol_5'] / (df['gk_vol_20'] + 1e-10)

        # Efficiency Ratio (Perry Kaufman) — mide si el precio se mueve en línea recta
        # ER = |net change| / sum(|daily changes|)  →  1 = tendencia perfecta, 0 = choppy
        for _er_period in [10, 20]:
            _net   = df[target_col].diff(_er_period).abs()
            _total = returns_series.abs().rolling(_er_period).sum()
            df[f'efficiency_ratio_{_er_period}'] = _net / (_total + 1e-10)

        # Overnight gap vs intraday return — presión institucional vs retail
        # Los gaps grandes indican flujo institucional que continúa
        if 'Open' in df.columns:
            df['gap_ret']       = (df['Open'] / df[target_col].shift(1) - 1)
            df['intraday_ret']  = (df[target_col] / df['Open'] - 1)
            df['gap_sum_5']     = df['gap_ret'].rolling(5).sum()
            df['gap_dominance'] = df['gap_ret'].abs() / (df['gap_ret'].abs() + df['intraday_ret'].abs() + 1e-10)

        # ── DRAWDOWN FEATURES ────────────────────────────────────────────
        # Critical for capturing how deep an asset is in a downtrend.
        # Models without these features systematically under-predict large bearish moves.
        for window in [10, 20, 60]:
            rolling_max = df[target_col].rolling(window, min_periods=5).max()
            df[f'drawdown_{window}d'] = (df[target_col] - rolling_max) / (rolling_max.abs() + 1e-10)

        # Consecutive losing days (momentum exhaustion signal)
        df['consec_down_days'] = (
            returns_series.lt(0).astype(int)
            .groupby((returns_series.lt(0) != returns_series.lt(0).shift()).cumsum())
            .cumsum()
        )
        # Dampen the streak to avoid extreme outlier values
        df['consec_down_days'] = df['consec_down_days'].clip(upper=15)

        # Recovery ratio: how much has price recovered from its N-day low
        for window in [20, 60]:
            rolling_min = df[target_col].rolling(window, min_periods=5).min()
            rolling_max2 = df[target_col].rolling(window, min_periods=5).max()
            df[f'recovery_{window}d'] = (df[target_col] - rolling_min) / (rolling_max2 - rolling_min + 1e-10)

        # Variables objetivo
        df['target_return'] = df[target_col].pct_change(prediction_horizon).shift(-prediction_horizon)
        df['target_direction'] = (df['target_return'] > 0).astype(int)
        df['target_price'] = df[target_col].shift(-prediction_horizon)

        # Features reducidas a ~25 indicadores estables con evidencia empírica.
        # Se eliminaron features ruidosas con alta varianza en muestras pequeñas:
        # hurst_, gk_vol_, gk_ratio, efficiency_ratio_, returns_skew_, returns_kurt_,
        # autocorr_, gap_sum_, gap_dominance, trend_consistency_, consec_down_days,
        # recovery_60d, mom_ (multicolineal con momentum_), ich_ (requiere 52 semanas)
        self.feature_names = [col for col in df.columns if col.startswith((
            'return_',       # lags de retorno 1,2,3,5,10d — señal de momentum
            'sma_',          # ratios SMA — tendencia
            'volatility_',   # volatilidad histórica 10,20d
            'rsi',           # RSI 14 — momentum/sobrecompra
            'macd',          # MACD — cruce de medias
            'bb_',           # Bollinger Bands — reversión a media
            'volume_',       # volumen relativo — confirmación
            'obv',           # OBV — acumulación/distribución
            'stoch_',        # Estocástico — momentum
            'adx',           # ADX — fuerza de tendencia
            'di_',           # DI+/DI- — dirección tendencia
            'momentum_',     # momentum multi-temporal 5,10,20d
            'vol_regime',    # régimen de volatilidad (bajo/alto)
            'vol_percentile', # percentil de vol actual vs histórica
            'bull_market',   # precio > SMA200 (régimen alcista)
            'sma_200_dist',  # distancia al SMA200
            'price_zscore',  # z-score precio vs media 20d
            'drawdown_',     # drawdown desde máximo (20d)
            'mfi',           # Money Flow Index
            'gap_ret',       # retorno gap overnight
            'intraday_ret',  # retorno intradía
            'recovery_20d',  # recuperación desde mínimo 20d (no 60d)
            # Tier 3.1: cross-asset macro features (yfinance VIX/DXY/TNX/SPY/BTC)
            'macro_',        # VIX close/chg, DXY chg, term_spread, US10Y chg
            'xa_',            # excess vs SPY, relative_strength, BTC dominance, alt_vs_btc
        ))]

        return df.dropna()
    
    def prepare_ml_data(self, df: pd.DataFrame, feature_cols: List[str],
                       target_col: str, seq_length: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos para modelos ML (salida 2D aplanada para RF/XGB/LGB).
        seq_length=5: ventana de 5 días reduce dimensionalidad (25 features × 5 = 125)
        vs la ventana anterior de 10 (25 × 10 = 250) para menos overfitting.
        """
        X, y = [], []

        for i in range(len(df) - seq_length):
            X.append(df[feature_cols].iloc[i:i+seq_length].values.flatten())
            y.append(df[target_col].iloc[i+seq_length])

        return np.array(X), np.array(y)

    def prepare_sequence_data(self, df: pd.DataFrame, feature_cols: List[str],
                              target_col: str, seq_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos en formato 3D para LSTM/GRU: (samples, seq_length, n_features)
        """
        data = df[feature_cols].values
        targets = df[target_col].values
        X_seq, y_seq = [], []
        for i in range(len(data) - seq_length):
            X_seq.append(data[i:i + seq_length])
            y_seq.append(targets[i + seq_length])
        return np.array(X_seq), np.array(y_seq)


class TraditionalMLModels:
    """Modelos tradicionales de Machine Learning"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_engineer = FeatureEngineer()
        self.results = {}
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                           task: str = 'regression') -> Dict:
        """
        Entrena modelo Random Forest
        """
        logger.info("Entrenando Random Forest...")
        
        if task == 'regression':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        
        # Feature importance
        feature_importance = dict(zip(
            [f'feature_{i}' for i in range(X_train.shape[1])],
            model.feature_importances_
        ))
        
        return {
            'model': model,
            'feature_importance': feature_importance
        }
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     task: str = 'regression') -> Dict:
        """
        Entrena modelo XGBoost
        """
        logger.info("Entrenando XGBoost...")
        
        if task == 'regression':
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        else:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        
        model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        
        return {
            'model': model,
            'feature_importance': dict(model.get_booster().get_fscore())
        }
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      task: str = 'regression') -> Dict:
        """
        Entrena modelo LightGBM
        """
        logger.info("Entrenando LightGBM...")
        
        if task == 'regression':
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        else:
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        
        model.fit(X_train, y_train)
        
        self.models['lightgbm'] = model
        
        return {
            'model': model
        }
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray,
                 task: str = 'regression') -> Dict:
        """
        Entrena modelo SVM
        """
        logger.info("Entrenando SVM...")
        
        # Escalar datos para SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['svm'] = scaler
        
        if task == 'regression':
            model = SVR(kernel='rbf', C=1.0, gamma='scale')
        else:
            model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        
        model.fit(X_train_scaled, y_train)
        
        self.models['svm'] = model
        
        return {
            'model': model,
            'scaler': scaler
        }
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray,
                       task: str = 'regression') -> pd.DataFrame:
        """
        Evalúa todos los modelos entrenados
        """
        results = []
        
        for name, model in self.models.items():
            logger.info(f"Evaluando {name}...")
            
            # Predecir
            if name == 'svm' and 'svm' in self.scalers:
                X_test_scaled = self.scalers['svm'].transform(X_test)
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
            
            if task == 'regression':
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'Modelo': name,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R²': r2
                })
                
                logger.info(f"  {name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            else:  # classification
                accuracy = accuracy_score(y_test, y_pred)
                
                results.append({
                    'Modelo': name,
                    'Accuracy': accuracy
                })
                
                logger.info(f"  {name} - Accuracy: {accuracy:.4f}")
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def predict(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """
        Realiza predicciones con el modelo especificado o el mejor modelo
        """
        if model_name and model_name in self.models:
            model = self.models[model_name]
            if model_name == 'svm' and 'svm' in self.scalers:
                X = self.scalers['svm'].transform(X)
            return model.predict(X)
        
        # Usar el mejor modelo (menor RMSE para regresión)
        if not self.results.empty:
            best_model = self.results.loc[self.results['RMSE'].idxmin(), 'Modelo']
            return self.predict(X, best_model)
        
        raise ValueError("No hay modelos entrenados")
    
    def save_models(self, path: str):
        """
        Guarda los modelos entrenados
        """
        for name, model in self.models.items():
            filepath = f"{path}/{name}_model.pkl"
            joblib.dump(model, filepath)
            logger.info(f"Modelo guardado: {filepath}")
    
    def load_models(self, path: str):
        """
        Carga modelos previamente entrenados
        """
        import glob
        model_files = glob.glob(f"{path}/*_model.pkl")

        for filepath in model_files:
            name = filepath.split('/')[-1].replace('_model.pkl', '')
            self.models[name] = joblib.load(filepath)
            logger.info(f"Modelo cargado: {name}")

    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray, n_trials: int = 30) -> dict:
        """
        Ajuste de hiperparámetros con Optuna para XGBoost y LightGBM.
        Solo se ejecuta si Optuna está disponible y hay suficientes datos.
        Devuelve los mejores parámetros encontrados para cada modelo.
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna no disponible. Usando hiperparámetros por defecto.")
            return {}
        if len(X) < 200:
            logger.info("Pocos datos para Optuna (<200). Usando hiperparámetros por defecto.")
            return {}

        from sklearn.model_selection import cross_val_score
        tscv = TimeSeriesSplit(n_splits=5, gap=5)  # gap=5 avoids data leakage between folds
        best_params = {}

        # Tune XGBoost (with L1/L2 regularization to prevent overfitting)
        logger.info(f"Optimizando XGBoost con Optuna ({n_trials} trials)...")
        def xgb_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
                'random_state': 42, 'n_jobs': -1,
            }
            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_root_mean_squared_error')
            return -scores.mean()

        study_xgb = optuna.create_study(direction='minimize', storage=None,
                                         sampler=optuna.samplers.TPESampler(seed=42))
        study_xgb.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=False)
        best_params['xgboost'] = study_xgb.best_params
        logger.info(f"XGBoost best RMSE: {study_xgb.best_value:.6f}")

        # Tune LightGBM (with L1/L2 regularization)
        logger.info(f"Optimizando LightGBM con Optuna ({n_trials} trials)...")
        def lgb_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
                'random_state': 42, 'verbose': -1,
            }
            model = lgb.LGBMRegressor(**params)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_root_mean_squared_error')
            return -scores.mean()

        study_lgb = optuna.create_study(direction='minimize', storage=None,
                                         sampler=optuna.samplers.TPESampler(seed=42))
        study_lgb.optimize(lgb_objective, n_trials=n_trials, show_progress_bar=False)
        best_params['lightgbm'] = study_lgb.best_params
        logger.info(f"LightGBM best RMSE: {study_lgb.best_value:.6f}")

        return best_params


class VolatilityPredictor:
    """Predictor especializado en volatilidad"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
    
    def calculate_realized_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calcula volatilidad realizada
        """
        returns = df['Close'].pct_change()
        realized_vol = returns.rolling(window).std() * np.sqrt(252)
        return realized_vol
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características específicas para predicción de volatilidad
        """
        df = df.copy()
        
        # Retornos
        df['returns'] = df['Close'].pct_change()
        
        # Volatilidad histórica
        for window in [5, 10, 20, 60]:
            df[f'hist_vol_{window}'] = df['returns'].rolling(window).std() * np.sqrt(252)
        
        # Retornos al cuadrado (varianza)
        df['returns_sq'] = df['returns'] ** 2
        
        # Rangos
        df['daily_range'] = (df['High'] - df['Low']) / df['Close']
        df['true_range'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                np.abs(df['High'] - df['Close'].shift(1)),
                np.abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['atr'] = df['true_range'].rolling(14).mean()
        
        # GARCH-like features
        df['abs_returns'] = np.abs(df['returns'])
        for lag in [1, 2, 3, 5]:
            df[f'abs_return_lag_{lag}'] = df['abs_returns'].shift(lag)
            df[f'return_sq_lag_{lag}'] = df['returns_sq'].shift(lag)
        
        # Volumen
        if 'Volume' in df.columns:
            df['volume_change'] = df['Volume'].pct_change()
            df['volume_ma'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma']
        
        # Target: volatilidad futura
        df['target_vol'] = df['hist_vol_20'].shift(-20)
        
        return df.dropna()
    
    def train(self, df: pd.DataFrame):
        """
        Entrena modelos de predicción de volatilidad
        """
        df_features = self.create_volatility_features(df)
        
        feature_cols = [col for col in df_features.columns if col.startswith(('hist_vol_', 'abs_return_', 'return_sq_', 'volume_', 'daily_range', 'atr'))]
        
        X = df_features[feature_cols].values
        y = df_features['target_vol'].values

        # Dividir temporalmente
        split_idx = int(len(X) * 0.8)
        X_train = self.scaler.fit_transform(X[:split_idx])
        X_test = self.scaler.transform(X[split_idx:])
        y_train, y_test = y[:split_idx], y[split_idx:]

        EMBARGO = 20
        if len(X_train) > EMBARGO + 10:
            X_train, y_train = X_train[:-EMBARGO], y_train[:-EMBARGO]

        # Entrenar modelos
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lgb': lgb.LGBMRegressor(n_estimators=100, verbose=-1)
        }
        
        for name, model in models.items():
            logger.info(f"Entrenando modelo de volatilidad: {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            logger.info(f"  RMSE: {rmse:.4f}")
            self.models[name] = model
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predice volatilidad con todos los modelos
        """
        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)
        
        return predictions
