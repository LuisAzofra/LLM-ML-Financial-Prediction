"""
Utilidades para manejo de datos financieros
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Tier 3.1: cache simple en memoria de series macro ───────────────────
# Las descargas de yfinance para VIX, DXY, ^TNX, etc., se reutilizan entre
# llamadas dentro del mismo proceso (Flask siempre vivo). Evita ~500ms/call.
_MACRO_CACHE: Dict[Tuple[str, str, str], pd.DataFrame] = {}

# Símbolos yfinance para series macro
_MACRO_SYMBOLS = {
    'vix':       '^VIX',     # VIX (CBOE Volatility Index)
    'dxy':       'DX-Y.NYB', # Dollar Index
    'us10y':     '^TNX',     # US 10-Year Treasury Yield (×10 → /10)
    'us2y':      '^IRX',     # US 13-Week T-Bill (proxy del front-end)
    'spy':       'SPY',      # S&P 500 ETF (benchmark stocks)
    'btc':       'BTC-USD',  # Bitcoin (referencia crypto)
    'eth':       'ETH-USD',  # Ethereum
}


def _fetch_macro_series(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Descarga una serie OHLCV vía yfinance con cache en memoria. Devuelve None si falla."""
    key = (symbol, start_date, end_date)
    if key in _MACRO_CACHE:
        return _MACRO_CACHE[key]
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(start=start_date, end=end_date, auto_adjust=False)
        if df is None or df.empty:
            _MACRO_CACHE[key] = None  # type: ignore
            return None
        # Normalizar índice: si es tz-aware → tz-naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        _MACRO_CACHE[key] = df[['Close']].rename(columns={'Close': 'close'})
        return _MACRO_CACHE[key]
    except Exception as e:
        logger.warning(f"yfinance fetch {symbol} failed: {e}")
        _MACRO_CACHE[key] = None  # type: ignore
        return None


def add_cross_asset_features(
    df: pd.DataFrame,
    symbol: str,
    asset_type: str = 'stock',
) -> pd.DataFrame:
    """
    Tier 3.1: enriquece `df` con features macro / cross-asset relevantes
    para `symbol`/`asset_type`. NO reemplaza nada: añade columnas con prefijos
    `macro_`, `xa_` que se incorporan al feature set ML existente.

    Stock features:
    - macro_vix_close, macro_vix_chg_5d
    - macro_dxy_chg_5d, macro_dxy_chg_20d
    - macro_term_spread (TNX − IRX, proxy del 10Y-2Y)
    - xa_excess_5d (retorno 5d del activo − retorno 5d SPY)
    - xa_relative_strength_20d (return_20d / spy_return_20d − 1)

    Crypto features:
    - macro_vix_close, macro_vix_chg_5d (risk-on/off proxy)
    - xa_btc_dominance_proxy (ratio BTC/(BTC+ETH) close por fecha)
    - xa_alt_vs_btc_5d (return_5d activo − return_5d BTC)

    Si una serie no se puede descargar (yfinance hiccup), la columna correspondiente
    se rellena con NaN y se imputa por forward-fill durante prepare_ml_data.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date   = (df.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    is_crypto = asset_type == 'crypto' or symbol.upper().endswith('-USD')

    def _align(macro_df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
        """Reindex macro_df sobre las fechas de df con asof backwards-fill."""
        if macro_df is None or macro_df.empty:
            return None
        s = macro_df['close'].reindex(df.index, method='ffill')
        return s

    # ── Features comunes (VIX como risk-on/off para todos) ─────────────
    vix_df = _fetch_macro_series(_MACRO_SYMBOLS['vix'], start_date, end_date)
    vix_close = _align(vix_df)
    if vix_close is not None:
        df['macro_vix_close']    = vix_close
        df['macro_vix_chg_5d']   = vix_close.pct_change(5)  * 100
        df['macro_vix_chg_20d']  = vix_close.pct_change(20) * 100

    # ── Stock features ─────────────────────────────────────────────────
    if not is_crypto:
        # DXY (Dollar)
        dxy_df = _fetch_macro_series(_MACRO_SYMBOLS['dxy'], start_date, end_date)
        dxy_close = _align(dxy_df)
        if dxy_close is not None:
            df['macro_dxy_chg_5d']  = dxy_close.pct_change(5)  * 100
            df['macro_dxy_chg_20d'] = dxy_close.pct_change(20) * 100

        # Term spread (^TNX − ^IRX) — proxy de 10Y-3M ≈ 10Y-2Y
        tnx_df = _fetch_macro_series(_MACRO_SYMBOLS['us10y'], start_date, end_date)
        irx_df = _fetch_macro_series(_MACRO_SYMBOLS['us2y'],  start_date, end_date)
        tnx_close = _align(tnx_df)
        irx_close = _align(irx_df)
        if tnx_close is not None and irx_close is not None:
            df['macro_term_spread'] = (tnx_close - irx_close)
        elif tnx_close is not None:
            df['macro_us10y_chg_20d'] = tnx_close.pct_change(20) * 100

        # Cross-asset vs SPY (sólo si NO es SPY mismo)
        if symbol.upper() not in ('SPY', '^GSPC'):
            spy_df = _fetch_macro_series(_MACRO_SYMBOLS['spy'], start_date, end_date)
            spy_close = _align(spy_df)
            if spy_close is not None:
                spy_ret_5d  = spy_close.pct_change(5)
                spy_ret_20d = spy_close.pct_change(20)
                close_col = df['Close'] if 'Close' in df.columns else df.get('close')
                if close_col is not None:
                    df['xa_excess_5d']           = (close_col.pct_change(5) - spy_ret_5d) * 100
                    df['xa_excess_20d']          = (close_col.pct_change(20) - spy_ret_20d) * 100
                    df['xa_relative_strength']   = (close_col.pct_change(20) / (spy_ret_20d.abs() + 1e-6))

    # ── Crypto features ────────────────────────────────────────────────
    else:
        btc_df = _fetch_macro_series(_MACRO_SYMBOLS['btc'], start_date, end_date)
        eth_df = _fetch_macro_series(_MACRO_SYMBOLS['eth'], start_date, end_date)
        btc_close = _align(btc_df)
        eth_close = _align(eth_df)

        # BTC dominance proxy: BTC / (BTC + ETH). True dominance también
        # incluye altcoins, pero BTC+ETH cubre ~75% del cap → buen proxy.
        if btc_close is not None and eth_close is not None and (btc_close > 0).all():
            denom = btc_close + eth_close
            df['xa_btc_dominance_proxy'] = btc_close / denom.where(denom > 0, np.nan)

        # Alt vs BTC strength (sólo si NO es BTC)
        if symbol.upper() != 'BTC-USD' and btc_close is not None:
            btc_ret_5d  = btc_close.pct_change(5)
            btc_ret_20d = btc_close.pct_change(20)
            close_col = df['Close'] if 'Close' in df.columns else df.get('close')
            if close_col is not None:
                df['xa_alt_vs_btc_5d']  = (close_col.pct_change(5)  - btc_ret_5d) * 100
                df['xa_alt_vs_btc_20d'] = (close_col.pct_change(20) - btc_ret_20d) * 100

    # Forward fill para huecos (mercados macro cierran fines de semana)
    macro_cols = [c for c in df.columns if c.startswith(('macro_', 'xa_'))]
    if macro_cols:
        df[macro_cols] = df[macro_cols].ffill().bfill()

    n_added = len(macro_cols)
    if n_added > 0:
        logger.info(f"Cross-asset features añadidas para {symbol} ({asset_type}): {n_added} columnas")
    return df


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
        
        # Rellenar valores nulos en otras columnas numéricas con interpolación.
        # limit_direction='forward' (no 'both'): rellenar hacia atrás usaría valores
        # FUTUROS (look-ahead bias) y contaminaría el backtest.
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='forward')
        # La interpolación forward no cubre NaN iniciales; se descartan esas filas
        # en vez de rellenarlas con datos del futuro (backfill).
        df = df.dropna(subset=list(numeric_cols))

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
                          method: str = 'minmax', fit_size: int = None) -> pd.DataFrame:
        """
        Normaliza las características especificadas.

        AVISO leakage: si se normaliza ANTES de partir en train/test, calcular
        las estadísticas sobre todo el df filtra información del test (look-ahead
        bias). Pasa fit_size = nº de filas de train para ajustar min/max o
        mean/std solo con df.iloc[:fit_size] y evitarlo.
        """
        df = df.copy()
        # Solo las primeras fit_size filas (train) definen las estadísticas; si es
        # None se mantiene el comportamiento original (todo el df) por compatibilidad.
        fit_df = df.iloc[:fit_size] if fit_size is not None else df

        for col in columns:
            if col not in df.columns:
                continue

            if method == 'minmax':
                min_val = fit_df[col].min()
                max_val = fit_df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val)
                self.scaler_params[col] = {'min': min_val, 'max': max_val, 'method': 'minmax'}

            elif method == 'zscore':
                mean_val = fit_df[col].mean()
                std_val = fit_df[col].std()
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
