"""
Utilidades para manejo de datos financieros
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Clase para procesar y preparar datos financieros"""
    
    def __init__(self):
        self.scaler_params = {}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia el dataframe eliminando valores nulos y duplicados
        """
        initial_rows = len(df)
        
        # Eliminar duplicados
        df = df.drop_duplicates()
        
        # Eliminar filas con valores nulos en columnas críticas
        critical_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
        
        # Rellenar valores nulos en otras columnas numéricas con interpolación
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
        
        final_rows = len(df)
        logger.info(f"Limpieza: {initial_rows} -> {final_rows} filas")
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade indicadores técnicos al dataframe
        """
        df = df.copy()
        
        # Medias móviles
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bandas de Bollinger
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Percent'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Volatilidad
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Volumen
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

        # ── ADDITIONAL ROBUST INDICATORS ──────────────────────────────────

        # Stochastic Oscillator %K / %D (14-period)
        if 'High' in df.columns and 'Low' in df.columns:
            low_14  = df['Low'].rolling(14).min()
            high_14 = df['High'].rolling(14).max()
            df['Stoch_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14 + 1e-10)
            df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

            # Williams %R
            df['Williams_R'] = -100 * (high_14 - df['Close']) / (high_14 - low_14 + 1e-10)

            # Commodity Channel Index (CCI, 20-period)
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            tp_sma = typical_price.rolling(20).mean()
            mean_deviation = typical_price.rolling(20).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True
            )
            df['CCI'] = (typical_price - tp_sma) / (0.015 * mean_deviation + 1e-10)

        # On-Balance Volume (OBV) — accumulation/distribution proxy
        if 'Volume' in df.columns:
            obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            df['OBV'] = obv
            df['OBV_SMA'] = obv.rolling(20).mean()
            df['OBV_Ratio'] = obv / (df['OBV_SMA'].abs() + 1e-10)

        # Price relative to 52-week high/low (momentum measure)
        df['High_52W'] = df['Close'].rolling(252, min_periods=50).max()
        df['Low_52W']  = df['Close'].rolling(252, min_periods=50).min()
        df['Pct_From_52W_High'] = (df['Close'] - df['High_52W']) / (df['High_52W'] + 1e-10)
        df['Pct_From_52W_Low']  = (df['Close'] - df['Low_52W'])  / (df['Low_52W']  + 1e-10)

        # Volatility regime indicator (current 20-day vol vs 1-year average)
        df['Vol_Regime'] = df['Volatility'] / (df['Volatility'].rolling(252, min_periods=60).mean() + 1e-10)

        # ── HIGH-VALUE FEATURES FOR ML PREDICTION ─────────────────────────

        # SMA_200: trend filter baseline (golden/death cross)
        df['SMA_200'] = df['Close'].rolling(window=200, min_periods=50).mean()

        # Multi-period momentum returns (critical for 5-day predictor)
        df['Return_5d']  = df['Close'].pct_change(5)
        df['Return_10d'] = df['Close'].pct_change(10)
        df['Return_20d'] = df['Close'].pct_change(20)

        # Distance from MAs normalized by ATR (mean-reversion / trend strength)
        df['Dist_SMA20_ATR']  = (df['Close'] - df['SMA_20'])  / (df['ATR'] + 1e-10)
        df['Dist_SMA50_ATR']  = (df['Close'] - df['SMA_50'])  / (df['ATR'] + 1e-10)
        df['Dist_SMA200_ATR'] = (df['Close'] - df['SMA_200']) / (df['ATR'] + 1e-10)

        # Average Directional Index (ADX) — trend strength, not direction
        if 'High' in df.columns and 'Low' in df.columns:
            _high  = df['High'].astype(float)
            _low   = df['Low'].astype(float)
            _close = df['Close'].astype(float)
            _tr    = pd.concat([
                _high - _low,
                (_high - _close.shift(1)).abs(),
                (_low  - _close.shift(1)).abs()
            ], axis=1).max(axis=1)
            _plus_dm_orig  = _high.diff().clip(lower=0)
            _minus_dm_orig = (-_low.diff()).clip(lower=0)
            # Fix: use originals for comparison to avoid sequential mutation bias
            _both     = (_plus_dm_orig > 0) & (_minus_dm_orig > 0)
            _plus_dm  = _plus_dm_orig.where(~(_both & (_minus_dm_orig >= _plus_dm_orig)), other=0)
            _minus_dm = _minus_dm_orig.where(~(_both & (_plus_dm_orig >= _minus_dm_orig)), other=0)
            _atr14    = _tr.rolling(14, min_periods=1).mean()
            _plus_di  = 100 * _plus_dm.rolling(14, min_periods=1).mean()  / (_atr14 + 1e-10)
            _minus_di = 100 * _minus_dm.rolling(14, min_periods=1).mean() / (_atr14 + 1e-10)
            _dx = 100 * (_plus_di - _minus_di).abs() / (_plus_di + _minus_di + 1e-10)
            df['ADX'] = _dx.rolling(14, min_periods=1).mean()

        # Money Flow Index (MFI) — volume-weighted RSI, catches divergence
        if 'Volume' in df.columns and 'High' in df.columns and 'Low' in df.columns:
            _tp  = (df['High'] + df['Low'] + df['Close']) / 3
            _rmf = _tp * df['Volume']
            _tp_diff = _tp.diff()
            _pos_mf = _rmf.where(_tp_diff > 0, 0.0).rolling(14).sum()
            _neg_mf = _rmf.where(_tp_diff < 0, 0.0).rolling(14).sum()
            df['MFI'] = 100 - (100 / (1 + _pos_mf / (_neg_mf + 1e-10)))

        # Chaikin Money Flow (CMF, 20-period) — buying/selling pressure
        if 'Volume' in df.columns and 'High' in df.columns and 'Low' in df.columns:
            _mf_mult = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / \
                       (df['High'] - df['Low'] + 1e-10)
            _mf_vol  = _mf_mult * df['Volume']
            df['CMF'] = _mf_vol.rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-10)

        # Rate of Change (ROC) — pure momentum
        df['ROC_5']  = df['Close'].pct_change(5)   * 100
        df['ROC_20'] = df['Close'].pct_change(20)  * 100

        logger.info("Indicadores técnicos añadidos")
        return df
    
    def normalize_features(self, df: pd.DataFrame, columns: List[str], 
                          method: str = 'minmax') -> pd.DataFrame:
        """
        Normaliza las características especificadas
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val)
                self.scaler_params[col] = {'min': min_val, 'max': max_val, 'method': 'minmax'}
                
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[col] = (df[col] - mean_val) / std_val
                self.scaler_params[col] = {'mean': mean_val, 'std': std_val, 'method': 'zscore'}
        
        return df
    
    def create_sequences(self, data: np.ndarray, seq_length: int, 
                        target_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea secuencias para modelos de series temporales
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length, target_idx])
        return np.array(X), np.array(y)
    
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.7, 
                   val_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """
        Divide los datos en train, validation y test
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return {
            'train': df.iloc[:train_end],
            'validation': df.iloc[train_end:val_end],
            'test': df.iloc[val_end:]
        }


def calculate_volatility_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula métricas de volatilidad
    """
    returns = df['Close'].pct_change().dropna()
    
    return {
        'annualized_volatility': returns.std() * np.sqrt(252),
        'daily_volatility': returns.std(),
        'max_daily_return': returns.max(),
        'min_daily_return': returns.min(),
        'value_at_risk_95': np.percentile(returns, 5),
        'value_at_risk_99': np.percentile(returns, 1),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis()
    }
