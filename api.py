"""
================================================================================
FINANCIAL AI - Flask API Server
Exposes the ML+LLM hybrid prediction system via REST endpoints
================================================================================
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import sys
import time
import warnings
import traceback
import json
import yfinance as yf

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import system modules
from data.data_loader import FinancialDataLoader
from utils.data_utils import DataProcessor
from data.news_fetcher import NewsFetcher
from models.pattern_detection.chart_patterns import PatternDetector
from models.ml_models.traditional_ml import TraditionalMLModels, FeatureEngineer
from models.ml_models.volatility_models import GARCHVolatilityModel
from agents.technical_agent.technical_analyst import TechnicalAnalystAgent
from agents.sentiment_agent.sentiment_analyst import SentimentAnalystAgent
from agents.risk_agent.risk_manager import RiskManagerAgent
from agents.manager_agent.portfolio_manager import PortfolioManagerAgent
from agents.base_agent import AgentCommunicationBus
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

# Optional LSTM/deep learning — IMPORT PEREZOSO
# TensorFlow añade ~700MB al arranque del proceso. En máquinas con swap
# saturado, eso impide cargar Qwen después. Lo importamos sólo si:
#   - LIGHTWEIGHT_MODE != '1' (modo completo) Y
#   - Se desactiva explícitamente con TFG_DISABLE_TF != '1'
DEEP_LEARNING_AVAILABLE = False
LSTMModel = None
if os.environ.get('TFG_LIGHTWEIGHT', '0') != '1' and os.environ.get('TFG_DISABLE_TF', '0') != '1':
    try:
        from models.ml_models.deep_learning import LSTMModel  # noqa: F401
        DEEP_LEARNING_AVAILABLE = True
        logger.info("TensorFlow/LSTM disponible")
    except Exception:
        DEEP_LEARNING_AVAILABLE = False
        logger.info("TensorFlow no disponible - se usarán solo modelos tradicionales")
else:
    logger.info("TensorFlow desactivado (lightweight mode o TFG_DISABLE_TF=1)")

# ──────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────
app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

TIMEFRAME_MAP = {
    '1m': 30,
    '3m': 90,
    '6m': 180,
    '1y': 365,
    '2y': 730,
    '3y': 1095,
    # Frontend timeframe pill aliases
    '5M': 30, '1H': 30, '12H': 90,
    '24H': 365, '1W': 365, '1M': 730, '1Y': 1095,
}

# Proveedor LLM por defecto — controlado por env var para poder elegir
# 'local' (Qwen2.5 vía transformers, sin Ollama) sin tener que tocar código.
DEFAULT_LLM_PROVIDER = os.environ.get('TFG_LLM_PROVIDER', 'ollama')
# Modo ligero: salta Optuna y reduce iteraciones de ML para que ML+LLM
# entren en RAM en máquinas saturadas de swap. Activar con TFG_LIGHTWEIGHT=1.
LIGHTWEIGHT_MODE = os.environ.get('TFG_LIGHTWEIGHT', '0') == '1'
logger.info(f"LLM provider default: {DEFAULT_LLM_PROVIDER}  |  lightweight: {LIGHTWEIGHT_MODE}")


# ──────────────────────────────────────────────
# Lightweight analysis (cabe en RAM con LLM cargado en máquinas saturadas)
# ──────────────────────────────────────────────
def run_lightweight_analysis(symbol: str, timeframe: str = '1y',
                              asset_type: str = 'stock',
                              llm_provider: str = 'local') -> dict:
    """
    Versión ligera de run_full_analysis. Usa el mismo flujo que demo_bot_llm.py:
      - Descarga datos + indicadores técnicos (sin GARCH).
      - ML proxy = retorno reciente + momentum (sin XGB/LGB/Optuna).
      - LLM Qwen2.5-0.5B local: analyze_sentiment + interpret_market_data.
      - Hybrid score 75% ML / 25% LLM.
    Devuelve el mismo shape JSON que run_full_analysis para el frontend.
    """
    from utils.llm_client import LLMClient
    days  = TIMEFRAME_MAP.get(timeframe, 365)
    end   = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    result = {
        'symbol': symbol, 'asset_type': asset_type, 'timeframe': timeframe,
        'start_date': start, 'end_date': end, 'status': 'running',
        'mode': 'lightweight',
    }

    loader = FinancialDataLoader()
    df = loader.download_data(symbol, start, end, asset_type=asset_type)
    if df is None or df.empty:
        result['status'] = 'error'
        result['error'] = f'No data for {symbol}'
        return result

    actual_symbol = df['Symbol'].iloc[0] if 'Symbol' in df.columns else symbol
    dp = DataProcessor()
    df = dp.clean_data(df)
    df = dp.add_technical_indicators(df)
    last = df.iloc[-1]

    result['data_info'] = {
        'records': len(df),
        'period_start': str(df.index[0].date()),
        'period_end':   str(df.index[-1].date()),
        'current_price': round(float(last['Close']), 4),
        'annualized_volatility': round(float(last.get('Volatility', 0) * 100), 2)
            if 'Volatility' in df.columns else None,
    }
    chart_df = df.tail(min(len(df), 500))
    result['price_chart'] = {
        'dates': [str(d.date()) for d in chart_df.index],
        'close': [round(float(v), 4) for v in chart_df['Close'].values],
        'sma_20': [round(float(v), 4) if not np.isnan(v) else None
                   for v in chart_df['SMA_20'].values] if 'SMA_20' in chart_df.columns else [],
        'sma_50': [round(float(v), 4) if not np.isnan(v) else None
                   for v in chart_df['SMA_50'].values] if 'SMA_50' in chart_df.columns else [],
    }
    result['garch'] = None  # skipped en modo light

    # ── ML proxy ───────────────────────────────────────────────────
    close = df['Close']
    r5  = float(close.pct_change(5).iloc[-1])  if len(close) > 5  else 0.0
    r20 = float(close.pct_change(20).iloc[-1]) if len(close) > 20 else 0.0
    pred_return = 0.5 * r5 + 0.5 * r20
    direction = 'UP' if pred_return > 0 else 'DOWN'
    ml_conf = round(0.5 + min(abs(pred_return) * 10, 0.4), 3)
    agreement = round(0.7 + min(abs(pred_return) * 5, 0.25), 3)
    ml_result = {
        'ensemble_prediction': round(pred_return, 6),
        'ensemble_confidence': ml_conf,
        'ensemble_direction':  direction,
        'model_agreement':     agreement,
        'best_model':          'lightweight_proxy',
        'horizon_days':        5,
        'mode':                'proxy_momentum',
    }
    result['ml'] = ml_result

    # ── Resúmenes para el LLM ──────────────────────────────────────
    rsi    = float(last.get('RSI', 50))    if 'RSI' in df.columns    else 50
    sma20  = float(last.get('SMA_20', 0))  if 'SMA_20' in df.columns else 0
    sma50  = float(last.get('SMA_50', 0))  if 'SMA_50' in df.columns else 0
    sma200 = float(last.get('SMA_200', 0)) if 'SMA_200' in df.columns else 0
    macd     = float(last.get('MACD', 0))        if 'MACD' in df.columns        else 0
    macd_sig = float(last.get('MACD_signal', 0)) if 'MACD_signal' in df.columns else 0
    volat = float(last.get('Volatility', 0) * 100) if 'Volatility' in df.columns else 0
    price = float(last['Close'])

    trend = []
    if sma20 and sma50:  trend.append('SMA20>SMA50' if sma20 > sma50 else 'SMA20<SMA50')
    if sma50 and sma200: trend.append('SMA50>SMA200' if sma50 > sma200 else 'SMA50<SMA200')
    trend_txt = ', '.join(trend) or 'n/a'

    technical_summary = (
        f"Precio actual ${price:,.2f}. RSI {rsi:.1f} "
        f"({'sobreventa' if rsi<30 else 'sobrecompra' if rsi>70 else 'neutral'}). "
        f"MACD {macd:.4f} vs signal {macd_sig:.4f}. Tendencia: {trend_txt}. "
        f"Vol anualizada {volat:.1f}%."
    )
    risk_summary = (
        f"Vol {volat:.1f}% anualizada. "
        f"Riesgo {'ALTO' if volat>40 else 'MEDIO' if volat>20 else 'BAJO'}."
    )
    ml_summary = (
        f"ML proxy: retorno esperado 5d {pred_return*100:+.2f}%, conf {ml_conf:.2f}, "
        f"acuerdo {agreement:.2f}, dirección {direction}."
    )

    # ── LLM ────────────────────────────────────────────────────────
    llm = LLMClient(provider=llm_provider)
    news_text = (
        f"{actual_symbol} cotiza a ${price:,.2f}. RSI {rsi:.0f}. "
        f"Tendencia {trend_txt}. Vol {volat:.0f}%. Retorno 20d {r20*100:+.1f}%."
    )
    sent = llm.analyze_sentiment(news_text)
    sent_p = sent.get('parsed', {}) if isinstance(sent, dict) else {}
    sent_label = sent_p.get('sentiment', 'neutral')
    sent_score = float(sent_p.get('score', 0) or 0)
    sent_conf  = float(sent_p.get('confidence', 0.5) or 0.5)

    sentiment_summary = (
        f"Sentimiento: {sent_label.upper()} (score {sent_score:+.2f}, conf {sent_conf:.2f}). "
        f"{str(sent_p.get('reasoning',''))[:160]}"
    )
    mkt = llm.interpret_market_data(
        technical_summary=technical_summary,
        sentiment_summary=sentiment_summary,
        risk_summary=risk_summary,
        ml_summary=ml_summary,
    )
    mkt_p = mkt.get('parsed', {}) if isinstance(mkt, dict) else {}
    raw_rec = str(mkt_p.get('recommendation', 'HOLD')).upper().strip()
    if '|' in raw_rec:  # placeholder de modelo pequeño
        raw_rec = 'HOLD'
    llm_conf = float(mkt_p.get('confidence', 0.5) or 0.5)

    # ── Estructura agents (lo que espera el frontend) ──────────────
    individual = {
        'technical': {
            'recommendation': ('COMPRAR' if pred_return > 0 else 'VENDER' if pred_return < 0 else 'MANTENER'),
            'confidence': round(ml_conf, 3),
            'reasoning': technical_summary,
        },
        'sentiment': {
            'recommendation': ('COMPRAR' if sent_score > 0.2 else 'VENDER' if sent_score < -0.2 else 'MANTENER'),
            'confidence': round(sent_conf, 3),
            'reasoning': sentiment_summary,
        },
        'risk': {
            'recommendation': 'MEDIO',
            'confidence': 0.7,
            'reasoning': risk_summary,
        },
        'ml_prediction': {
            'recommendation': direction,
            'confidence': ml_conf,
            'reasoning': ml_summary,
        },
    }
    REC_MAP = {'BUY': 'COMPRAR', 'SELL': 'VENDER', 'HOLD': 'MANTENER',
               'WAIT': 'ESPERAR', 'STRONG_BUY': 'COMPRAR_FUERTE', 'STRONG_SELL': 'VENDER_FUERTE'}
    final_dec = REC_MAP.get(raw_rec, raw_rec)
    result['agents'] = {
        'final_decision':   final_dec,
        'final_confidence': round(llm_conf, 3),
        'final_reasoning':  str(mkt_p.get('reasoning', ''))[:600],
        'individual':       individual,
    }

    # ── Hybrid score ───────────────────────────────────────────────
    ml_signal = max(-1.0, min(1.0, pred_return * 100))
    DEC_TO_SIG = {'COMPRAR': 0.7, 'COMPRAR_FUERTE': 1.0, 'VENDER': -0.7,
                  'VENDER_FUERTE': -1.0, 'MANTENER': 0.0, 'ESPERAR': 0.0}
    llm_signal = DEC_TO_SIG.get(final_dec, 0.0)
    hybrid_score = 0.75 * ml_signal * ml_conf + 0.25 * llm_signal * llm_conf
    hybrid_conf  = 0.75 * ml_conf + 0.25 * llm_conf
    disagree = (ml_signal > 0 and llm_signal < 0) or (ml_signal < 0 and llm_signal > 0)
    if disagree:
        hybrid_rec = 'MANTENER'
    elif hybrid_score > 0.4:
        hybrid_rec = 'COMPRAR'
    elif hybrid_score < -0.4:
        hybrid_rec = 'VENDER'
    else:
        hybrid_rec = 'MANTENER'
    result['hybrid'] = {
        'score':              round(hybrid_score, 4),
        'confidence':         round(hybrid_conf, 3),
        'recommendation':     hybrid_rec,
        'direction_conflict': bool(disagree),
        'ml_signal':          round(ml_signal, 3),
        'llm_signal':         round(llm_signal, 3),
    }
    result['status'] = 'success'
    result['summary'] = {
        'recommendation': hybrid_rec,
        'hybrid_score':   result['hybrid']['score'],
        'confidence':     result['hybrid']['confidence'],
        'ml_direction':   direction,
        'agent_decision': final_dec,
    }
    return result


# ──────────────────────────────────────────────
# Core analysis function
# ──────────────────────────────────────────────
def run_full_analysis(symbol: str, timeframe: str = '1y',
                      asset_type: str = 'stock',
                      use_llm: bool = True,
                      llm_provider: str = 'ollama') -> dict:
    """
    Runs the complete ML + LLM hybrid analysis pipeline for a given asset.
    Returns a structured dict with all results.
    """
    days = TIMEFRAME_MAP.get(timeframe, 365)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    result = {
        'symbol': symbol,
        'asset_type': asset_type,
        'timeframe': timeframe,
        'start_date': start_date,
        'end_date': end_date,
        'status': 'running',
    }

    # ── 1. Load data ──────────────────────────
    logger.info(f"[1/6] Loading data for {symbol} ({asset_type}) from {start_date} to {end_date}")
    loader = FinancialDataLoader()
    df = loader.download_data(symbol, start_date, end_date, asset_type=asset_type)

    if df is None or df.empty:
        result['status'] = 'error'
        result['error'] = f'No data found for {symbol}. Check the ticker symbol.'
        return result

    # Resolve actual symbol (may have been normalized for crypto)
    actual_symbol = df['Symbol'].iloc[0] if 'Symbol' in df.columns else symbol

    processor = DataProcessor()
    df = processor.clean_data(df)
    df = processor.add_technical_indicators(df)

    result['data_info'] = {
        'records': len(df),
        'period_start': str(df.index[0].date()),
        'period_end': str(df.index[-1].date()),
        'current_price': round(float(df['Close'].iloc[-1]), 4),
        'annualized_volatility': round(float(df['Volatility'].iloc[-1] * 100), 2) if 'Volatility' in df.columns else None,
    }

    # Price chart data (send last 200 data points max)
    chart_df = df.tail(min(len(df), 500))
    result['price_chart'] = {
        'dates': [str(d.date()) for d in chart_df.index],
        'close': [round(float(v), 4) for v in chart_df['Close'].values],
        'sma_20': [round(float(v), 4) if not np.isnan(v) else None
                   for v in chart_df['SMA_20'].values] if 'SMA_20' in chart_df.columns else [],
        'sma_50': [round(float(v), 4) if not np.isnan(v) else None
                   for v in chart_df['SMA_50'].values] if 'SMA_50' in chart_df.columns else [],
    }

    # ── 2. GARCH volatility ───────────────────
    logger.info(f"[2/6] Fitting GARCH volatility model")
    garch_result = None
    try:
        returns = df['Close'].pct_change().dropna().values
        if len(returns) >= 100:
            garch = GARCHVolatilityModel(p=1, q=1, model_type='GARCH', distribution='t')
            metrics_g = garch.fit(returns, disp='off')
            forecast_g = garch.forecast(horizon=5)
            garch_result = {
                'aic': round(float(metrics_g['aic']), 2),
                'bic': round(float(metrics_g['bic']), 2),
                'forecast_volatility': round(float(forecast_g.forecast_value * 100), 4),
                'ci_lower': round(float(forecast_g.confidence_interval[0] * 100), 4),
                'ci_upper': round(float(forecast_g.confidence_interval[1] * 100), 4),
            }
    except Exception as e:
        logger.warning(f"GARCH error: {e}")
    result['garch'] = garch_result

    # ── 3. ML models ──────────────────────────
    logger.info(f"[3/6] Training ML models")
    ml_result = _train_and_predict_ml(df, asset_type=asset_type, symbol=symbol)
    result['ml'] = ml_result

    # ── 4. Multi-agent analysis ──────────────
    logger.info(f"[4/6] Running multi-agent analysis")
    ml_pred = {
        'predicted_return': ml_result.get('ensemble_prediction', 0),
        'confidence': ml_result.get('ensemble_confidence', 0.5),
        **ml_result,  # pass full ml_result so portfolio manager can use all metrics
    }
    agent_result = _run_agents(df, actual_symbol, ml_pred, use_llm, llm_provider)
    result['agents'] = agent_result

    # ── 5. Hybrid score ──────────────────────
    logger.info(f"[5/6] Computing hybrid ML+LLM score")
    hybrid = _compute_hybrid_score(ml_result, agent_result)
    result['hybrid'] = hybrid

    # ── 6. Summary ───────────────────────────
    logger.info(f"[6/6] Generating summary")
    result['status'] = 'success'
    result['summary'] = {
        'recommendation': hybrid['recommendation'],
        'hybrid_score': hybrid['score'],
        'confidence': hybrid['confidence'],
        'ml_direction': ml_result.get('ensemble_direction', 'N/A'),
        'agent_decision': agent_result.get('final_decision', 'N/A'),
    }

    return result


# ──────────────────────────────────────────────
# ML training & prediction
# ──────────────────────────────────────────────
def _train_and_predict_ml(df: pd.DataFrame, asset_type: str = 'stock',
                           prediction_horizon: int = None,
                           symbol: str = '') -> dict:
    """
    Train ML models (RF, XGBoost, LightGBM, optional LSTM) and produce a
    stacking ensemble prediction on the latest data.

    Improvements:
    - Walk-forward cross-validation (TimeSeriesSplit) for honest metrics
    - Stacking meta-learner (Ridge) instead of simple weighted average
    - Extended features: regime detection, volatility percentile, multi-tf momentum
    - Feature selection: top-K features by XGBoost importance to reduce noise
    - Confidence calibration: avg R² + model agreement, not just best R²
    - Prediction horizon matches evaluation horizon (no more apples vs oranges)
    - Per-asset-type hyperparameters
    - Optional LSTM as 4th model
    """
    try:
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # ── Asset-type specific hyperparameters ──────────────────────────
        if asset_type == 'crypto':
            default_horizon = 7
            rf_params   = dict(n_estimators=200, max_depth=5, min_samples_split=10,
                               random_state=42, n_jobs=-1)
            xgb_params  = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                               reg_lambda=1.0, random_state=42, n_jobs=-1)
            lgb_params  = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                               min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                               random_state=42, verbose=-1)
            hgb_params  = dict(max_iter=200, max_depth=4, learning_rate=0.05,
                               l2_regularization=1.0, random_state=42)
        else:
            default_horizon = 5
            rf_params   = dict(n_estimators=300, max_depth=8, min_samples_split=8,
                               random_state=42, n_jobs=-1)
            xgb_params  = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                               reg_lambda=1.0, random_state=42, n_jobs=-1)
            lgb_params  = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                               min_child_samples=15, reg_alpha=0.1, reg_lambda=1.0,
                               random_state=42, verbose=-1)
            hgb_params  = dict(max_iter=300, max_depth=5, learning_rate=0.05,
                               l2_regularization=1.0, random_state=42)

        # Prediction horizon: use caller-supplied value if provided, else asset default
        if prediction_horizon is None:
            prediction_horizon = default_horizon
        # Cap at 90 days — beyond that ML signal-to-noise degrades too much
        prediction_horizon = max(1, min(prediction_horizon, 90))

        # ── Feature engineering ──────────────────────────────────────────
        # Tier 3.1: enriquecer con cross-asset macro ANTES de create_features
        # para que los nuevos prefijos `macro_`/`xa_` se incluyan automáticamente
        # en self.feature_names si están listados en la tupla de prefijos
        # aceptados (ver traditional_ml.py:create_features).
        try:
            from utils.data_utils import add_cross_asset_features as _add_xa
            df = _add_xa(df, symbol=symbol or 'unknown', asset_type=asset_type)
        except Exception as xa_err:
            logger.warning(f"Cross-asset features skipped: {xa_err}")

        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.create_features(df, prediction_horizon=prediction_horizon)
        feature_cols = feature_engineer.feature_names

        # Inference window: recompute features with prediction_horizon=1 so the last rows
        # of df are NOT dropped. df_features loses the last `prediction_horizon` rows
        # (target_return is NaN there), which means df_features.iloc[-10:] reflects
        # features from `prediction_horizon` days ago — causing the model to predict
        # the PAST return, not the future. Using prediction_horizon=1 keeps all but the
        # final row, giving us today's feature values for forward-looking prediction.
        _fe_infer = FeatureEngineer()
        _df_infer = _fe_infer.create_features(df, prediction_horizon=1)

        if len(df_features) < 80:
            return {'error': 'Not enough data for ML training', 'models': []}

        X, y = feature_engineer.prepare_ml_data(df_features, feature_cols, 'target_return')

        # ── Feature selection FIRST: reduce dimensionality before tuning ──
        # Rationale: Optuna and final models should share the same feature space.
        # Having 500+ features (50 indicators × 10 time steps) causes overfitting.
        # We select top features by a quick XGBoost importance ranking.
        n_features_total = X.shape[1]
        selected_indices = np.arange(n_features_total)  # default: keep all
        if n_features_total > 40 and len(X) >= 100:
            try:
                quick_xgb = xgb.XGBRegressor(
                    n_estimators=200, max_depth=3, learning_rate=0.05,
                    subsample=0.7, colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=1.0,
                    random_state=42, n_jobs=-1
                )
                quick_xgb.fit(X, y)
                importances = quick_xgb.feature_importances_
                # Reducir de 60 → 30 features para menos overfitting
                n_keep = min(30, max(15, n_features_total // 3))
                selected_indices = np.argsort(importances)[-n_keep:]
                X = X[:, selected_indices]
                logger.info(f"Feature selection: {n_features_total} → {len(selected_indices)} features")
            except Exception as fs_err:
                logger.warning(f"Feature selection skipped: {fs_err}")
                selected_indices = np.arange(n_features_total)

        # ── Optuna hyperparameter tuning on REDUCED feature set ───────────
        ml_models_obj = TraditionalMLModels()
        if len(X) > 200 and not LIGHTWEIGHT_MODE:
            tuned = ml_models_obj.tune_hyperparameters(X, y, n_trials=25)
            if 'xgboost' in tuned:
                tuned['xgboost'].update({'random_state': 42, 'n_jobs': -1})
                xgb_params = tuned['xgboost']
            if 'lightgbm' in tuned:
                tuned['lightgbm'].update({'random_state': 42, 'verbose': -1})
                lgb_params = tuned['lightgbm']
        if LIGHTWEIGHT_MODE:
            # Reducir #estimators para minimizar pico de RAM al entrenar
            # 4 modelos en serie + Qwen-0.5B cargado en memoria.
            for _p in (rf_params, xgb_params, lgb_params):
                if 'n_estimators' in _p:
                    _p['n_estimators'] = min(_p['n_estimators'], 100)
            if 'max_iter' in hgb_params:
                hgb_params['max_iter'] = min(hgb_params['max_iter'], 100)
            logger.info("LIGHTWEIGHT_MODE: Optuna OFF · n_estimators=100")

        # ── Walk-forward cross-validation for honest metrics ─────────────
        # gap=15: embargo 5 bars + sequence length 10 — evita leakage de overlapping labels
        n_splits = max(2, min(5, len(X) // 40))
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=15)
        cv_metrics = {'random_forest': [], 'xgboost': [], 'lightgbm': [], 'hist_gradient': []}
        # Tier 2.1: recolectar OOF predictions del XGB (proxy del ensemble) para
        # fit isotónico posterior.
        oof_xgb = np.full(len(X), np.nan, dtype=float)
        # Tier 2.2: XGBClassifier direccional con scale_pos_weight (clase
        # imbalanceada en mercados alcistas). Sus probabilidades OOF son el
        # input PREFERENTE para el calibrador isotónico — son ya P(positive)
        # nativas, sin necesidad de mapear scores arbitrarios a P. Si el
        # classifier resulta NO accepted, fallback a oof_xgb regressor.
        oof_xgb_dir = np.full(len(X), np.nan, dtype=float)
        # Etiquetas binarias para el classifier
        y_dir = (y > 0).astype(int)
        # scale_pos_weight global (se ajusta por fold abajo)
        n_pos = int(y_dir.sum()); n_neg = int(len(y_dir) - n_pos)
        # Hiperparámetros del classifier (más conservadores que regressor)
        xgb_dir_params = dict(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, eval_metric='logloss',
        )

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            rf_fold  = RandomForestRegressor(**rf_params).fit(X_tr, y_tr)
            xgb_fold = xgb.XGBRegressor(**xgb_params).fit(X_tr, y_tr)
            lgb_fold = lgb.LGBMRegressor(**lgb_params).fit(X_tr, y_tr)
            hgb_fold = HistGradientBoostingRegressor(**hgb_params).fit(X_tr, y_tr)

            for name, model in [('random_forest', rf_fold), ('xgboost', xgb_fold),
                                 ('lightgbm', lgb_fold), ('hist_gradient', hgb_fold)]:
                preds = model.predict(X_te)
                cv_metrics[name].append({
                    'rmse': float(np.sqrt(mean_squared_error(y_te, preds))),
                    'mae':  float(mean_absolute_error(y_te, preds)),
                    'r2':   float(r2_score(y_te, preds)),
                })
            # Snapshot OOF predictions del XGB regressor (Tier 2.1)
            oof_xgb[test_idx] = xgb_fold.predict(X_te)

            # Tier 2.2: classifier direccional con scale_pos_weight per-fold
            try:
                y_tr_dir = (y_tr > 0).astype(int)
                n_pos_tr = int(y_tr_dir.sum()); n_neg_tr = int(len(y_tr_dir) - n_pos_tr)
                spw = (n_neg_tr / max(n_pos_tr, 1)) if n_pos_tr > 0 else 1.0
                xgb_dir_fold = xgb.XGBClassifier(
                    scale_pos_weight=spw, **xgb_dir_params
                ).fit(X_tr, y_tr_dir)
                oof_xgb_dir[test_idx] = xgb_dir_fold.predict_proba(X_te)[:, 1]
            except Exception as ce:
                logger.debug(f"XGB classifier OOF fold failed: {ce}")

        # Average CV metrics
        model_results = []
        for name, folds in cv_metrics.items():
            model_results.append({
                'name': name,
                'rmse': round(float(np.mean([f['rmse'] for f in folds])), 6),
                'mae':  round(float(np.mean([f['mae']  for f in folds])), 6),
                'r2':   round(float(np.mean([f['r2']   for f in folds])), 6),
            })

        # ── Train FINAL models on ALL data ───────────────────────────────
        rf_final  = RandomForestRegressor(**rf_params).fit(X, y)
        xgb_final = xgb.XGBRegressor(**xgb_params).fit(X, y)
        lgb_final = lgb.LGBMRegressor(**lgb_params).fit(X, y)
        hgb_final = HistGradientBoostingRegressor(**hgb_params).fit(X, y)
        # Tier 2.2: classifier final sobre todo X (para inferencia).
        # Si los OOF del classifier son válidos, este será el modelo de
        # P(direction) usado en inferencia para alimentar el calibrador.
        xgb_dir_final = None
        try:
            spw_final = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
            xgb_dir_final = xgb.XGBClassifier(
                scale_pos_weight=spw_final, **xgb_dir_params
            ).fit(X, y_dir)
        except Exception as ce:
            logger.warning(f"XGB classifier final fit failed: {ce}")

        base_models = [
            ('random_forest',  rf_final),
            ('xgboost',        xgb_final),
            ('lightgbm',       lgb_final),
            ('hist_gradient',  hgb_final),
        ]

        # ── Stacking ensemble (meta-learner) ─────────────────────────────
        # Out-of-fold predictions for Ridge meta-learner
        use_stacking = len(X) >= 100 and n_splits >= 2
        meta = None
        oof_matrix = None
        if use_stacking:
            try:
                oof_preds_list = []
                for _, model in base_models:
                    oof = cross_val_predict(model.__class__(**model.get_params()),
                                            X, y, cv=tscv)
                    oof_preds_list.append(oof)
                oof_matrix = np.column_stack(oof_preds_list)
                meta = Ridge(alpha=1.0).fit(oof_matrix, y)
            except Exception as stack_err:
                logger.warning(f"Stacking failed ({stack_err}), falling back to weighted average")
                use_stacking = False

        # ── Optional LSTM (4th model) ────────────────────────────────────
        lstm_pred_val = None
        lstm_metrics  = None
        if DEEP_LEARNING_AVAILABLE and len(df_features) > 150:
            try:
                seq_len = 30
                X_seq, y_seq = feature_engineer.prepare_sequence_data(
                    df_features, feature_cols, 'target_return', seq_length=seq_len
                )
                if len(X_seq) > 60:
                    n_features = X_seq.shape[2]
                    lstm_model = LSTMModel(seq_length=seq_len, n_features=n_features)
                    lstm_model.build_model(units=[64, 32], dropout_rate=0.2, learning_rate=0.001)
                    split_lstm = int(len(X_seq) * 0.8)
                    from tensorflow.keras.callbacks import EarlyStopping
                    es = EarlyStopping(patience=5, restore_best_weights=True)
                    lstm_model.model.fit(
                        X_seq[:split_lstm], y_seq[:split_lstm],
                        validation_data=(X_seq[split_lstm:], y_seq[split_lstm:]),
                        epochs=30, batch_size=32, callbacks=[es], verbose=0
                    )
                    lstm_val_preds = lstm_model.model.predict(X_seq[split_lstm:], verbose=0).flatten()
                    lstm_rmse = float(np.sqrt(mean_squared_error(y_seq[split_lstm:], lstm_val_preds)))
                    lstm_r2   = float(r2_score(y_seq[split_lstm:], lstm_val_preds))
                    lstm_mae  = float(mean_absolute_error(y_seq[split_lstm:], lstm_val_preds))

                    # Predict on last seq_len rows — use _df_infer (current features, not stale)
                    _lstm_src = _df_infer if all(c in _df_infer.columns for c in feature_cols) else df_features
                    latest_seq = _lstm_src[feature_cols].iloc[-seq_len:].values
                    lstm_pred_val = float(lstm_model.model.predict(
                        latest_seq.reshape(1, seq_len, n_features), verbose=0
                    ).flatten()[0])

                    lstm_metrics = {'name': 'lstm', 'rmse': round(lstm_rmse, 6),
                                    'mae': round(lstm_mae, 6), 'r2': round(lstm_r2, 6)}
                    model_results.append(lstm_metrics)
                    logger.info(f"LSTM: RMSE={lstm_rmse:.6f}, R²={lstm_r2:.6f}")
            except Exception as lstm_err:
                logger.warning(f"LSTM skipped: {lstm_err}")

        # ── Final prediction via stacking (or weighted avg fallback) ────────
        # Use the inference df (prediction_horizon=1) so we predict from TODAY's features,
        # not from features that are `prediction_horizon` days stale.
        _infer_rows = _df_infer[feature_cols].iloc[-10:] if all(c in _df_infer.columns for c in feature_cols) else df_features[feature_cols].iloc[-10:]
        latest_flat_full = _infer_rows.values.flatten().reshape(1, -1)
        # Apply the same feature selection used during training
        latest_flat = latest_flat_full[:, selected_indices] if len(selected_indices) < latest_flat_full.shape[1] else latest_flat_full
        base_preds_latest = np.array([
            rf_final.predict(latest_flat)[0],
            xgb_final.predict(latest_flat)[0],
            lgb_final.predict(latest_flat)[0],
            hgb_final.predict(latest_flat)[0],
        ])

        if use_stacking and meta is not None:
            stacked_pred = float(meta.predict(base_preds_latest.reshape(1, -1))[0])
            if lstm_pred_val is not None:
                ensemble_pred = round((stacked_pred + lstm_pred_val) / 2, 6)
            else:
                ensemble_pred = round(stacked_pred, 6)
        else:
            # Weighted average fallback — performance-weighted by inverse RMSE
            rmse_map = {mr['name']: mr['rmse'] for mr in model_results}
            base_names = ['random_forest', 'xgboost', 'lightgbm', 'hist_gradient']
            raw_w = np.array([1.0 / (rmse_map.get(n, 1.0) + 1e-8) for n in base_names])
            all_preds = base_preds_latest.copy()
            if lstm_pred_val is not None:
                # Give LSTM weight proportional to its performance vs best base model
                lstm_rmse = next((mr['rmse'] for mr in model_results if mr['name'] == 'lstm'), 1.0)
                lstm_w = 1.0 / (lstm_rmse + 1e-8)
                raw_w = np.append(raw_w, lstm_w)
                all_preds = np.append(all_preds, lstm_pred_val)
            weights = raw_w / raw_w.sum()
            ensemble_pred = round(float(np.dot(weights, all_preds)), 6)

        # Individual predictions for display
        predictions = {
            'random_forest': round(float(base_preds_latest[0]), 6),
            'xgboost':       round(float(base_preds_latest[1]), 6),
            'lightgbm':      round(float(base_preds_latest[2]), 6),
            'hist_gradient': round(float(base_preds_latest[3]), 6),
        }
        if lstm_pred_val is not None:
            predictions['lstm'] = round(lstm_pred_val, 6)

        # ── Confidence & direction ───────────────────────────────────────
        # Use AVERAGE R² across all models (not best), to avoid overestimating
        avg_r2 = float(np.mean([mr['r2'] for mr in model_results]))
        # Calibrated formula: avg_r2=0 → 30%, avg_r2=0.5 → 55%, avg_r2=1.0 → 80%
        # Negative R² models are penalized further
        if avg_r2 >= 0:
            base_conf = 0.30 + avg_r2 * 0.50
        else:
            base_conf = max(0.10, 0.30 + avg_r2 * 0.20)  # softer penalty for slight negatives

        # Penalize when base models disagree on direction (reduces false confidence)
        pred_signs = [int(np.sign(p)) for p in base_preds_latest]
        majority_sign = int(np.sign(sum(pred_signs)))
        agreement_ratio = sum(1 for s in pred_signs if s == majority_sign) / len(pred_signs)
        if agreement_ratio < 0.67:  # models point in different directions
            base_conf *= 0.70
            logger.info(f"Models disagree on direction (agreement={agreement_ratio:.0%}), reducing confidence")

        # Penalize for long prediction horizons (harder to predict)
        if prediction_horizon > 30:
            horizon_penalty = max(0.70, 1.0 - (prediction_horizon - 30) * 0.003)
            base_conf *= horizon_penalty

        confidence = round(max(0.10, min(0.85, base_conf)), 3)

        # ── Tier 2.1 + 2.2: refinamiento con calibración isotónica ───────
        # Estrategia en cascada:
        #   1. Preferir OOF probabilities del XGBClassifier direccional
        #      (Tier 2.2): probabilidades nativas, mejor reliability_corr.
        #   2. Si el classifier no produjo OOF válidas, fallback a OOF
        #      del XGBRegressor (Tier 2.1).
        #   3. Si ningún calibrador pasa el gate (Brier < baseline AND
        #      reliability_corr ≥ 0.5), fallback a confianza legacy.
        calibration_info = None
        try:
            from utils.calibration import (
                fit_calibrator_from_oof, combine_confidence, calibrator_path,
            )

            cal = None
            cal_metrics = None
            cal_source = None
            score_for_inference = None

            # 1) Intento con OOF probabilidades del classifier (Tier 2.2)
            if (xgb_dir_final is not None
                    and not np.isnan(oof_xgb_dir).all()):
                cal_clf, m_clf = fit_calibrator_from_oof(
                    oof_xgb_dir, y, name='direction_clf',
                )
                if cal_clf is not None:
                    cal = cal_clf
                    cal_metrics = m_clf
                    cal_source = 'direction_classifier'
                    # Score para inferencia: P del classifier final sobre fila latest
                    try:
                        p_clf = float(xgb_dir_final.predict_proba(latest_flat)[0, 1])
                        score_for_inference = p_clf
                    except Exception:
                        # latest_flat aún no definido en este punto; lo asignamos abajo
                        # cuando el bloque ya se ejecute después; este try captura el
                        # NameError silenciosamente y forzamos fallback al regressor.
                        cal = None

            # 2) Fallback: OOF del regressor (Tier 2.1)
            if cal is None:
                cal_reg, m_reg = fit_calibrator_from_oof(
                    oof_xgb, y, name='regressor_proxy',
                )
                if cal_reg is not None:
                    cal = cal_reg
                    cal_metrics = m_reg
                    cal_source = 'regressor_proxy'
                    score_for_inference = float(ensemble_pred)
                else:
                    cal_metrics = m_reg

            if cal is not None and score_for_inference is not None:
                p_cal = float(cal.predict_proba(np.array([score_for_inference]))[0])
                conf_cal = combine_confidence(p_cal, agreement_ratio, prediction_horizon)
                calibration_info = {
                    'used':                  True,
                    'source':                cal_source,
                    'p_positive_calibrated': round(p_cal, 4),
                    'brier_score':           round(cal_metrics.brier_score, 4),
                    'brier_baseline':        round(cal_metrics.base_rate * (1 - cal_metrics.base_rate), 4),
                    'reliability_corr':      round(cal_metrics.reliability_correlation, 3),
                    'n_samples':             cal_metrics.n_samples,
                    'confidence_legacy':     round(confidence, 3),
                    'confidence_calibrated': round(conf_cal, 3),
                }
                logger.info(
                    f"Calibration ON [{cal_source}]  Brier={cal_metrics.brier_score:.3f} "
                    f"(base {calibration_info['brier_baseline']:.3f}) "
                    f"corr={cal_metrics.reliability_correlation:.2f}  "
                    f"score={score_for_inference:.4f} → p_cal={p_cal:.3f}  "
                    f"conf {confidence:.3f} → {conf_cal:.3f}"
                )
                confidence = round(conf_cal, 3)
            else:
                calibration_info = {
                    'used':              False,
                    'reason':            'rejected_by_filters',
                    'brier_score':       round(cal_metrics.brier_score, 4) if cal_metrics else None,
                    'brier_baseline':    round(cal_metrics.base_rate * (1 - cal_metrics.base_rate), 4) if cal_metrics else None,
                    'reliability_corr':  round(cal_metrics.reliability_correlation, 3) if cal_metrics else None,
                    'n_samples':         cal_metrics.n_samples if cal_metrics else 0,
                }
                logger.info(
                    f"Calibration OFF (rejected) → fallback legacy conf={confidence:.3f}"
                )
        except Exception as cal_err:
            calibration_info = {'used': False, 'error': str(cal_err)}
            logger.warning(f"Calibration error → fallback legacy: {cal_err}")

        direction  = 'SUBE' if ensemble_pred > 0.001 else 'BAJA' if ensemble_pred < -0.001 else 'LATERAL'

        # ── Feature importance (XGBoost top-10) — feature-selection aware ─
        importance_vals = xgb_final.feature_importances_
        n_base = len(feature_cols)
        # seq_len used to flatten: original total was n_features_total = n_base * seq_len
        seq_len_orig = n_features_total // n_base if n_base > 0 else 1
        # Map each selected position back to its original feature name and aggregate
        name_importance = {name: 0.0 for name in feature_cols}
        for sel_pos, imp in zip(selected_indices, importance_vals):
            if seq_len_orig > 1:
                orig_name = feature_cols[sel_pos % n_base]
            else:
                orig_name = feature_cols[min(int(sel_pos), n_base - 1)]
            name_importance[orig_name] += float(imp)
        top10 = sorted(name_importance.items(), key=lambda x: -x[1])[:10]
        top_features = [{'name': f, 'importance': round(v, 4)} for f, v in top10 if v > 0]

        return {
            'models': model_results,
            'predictions': predictions,
            'ensemble_prediction': ensemble_pred,
            'ensemble_direction': direction,
            'ensemble_confidence': confidence,
            'n_features': len(feature_cols),
            'n_features_selected': int(len(selected_indices)),
            'prediction_horizon_days': prediction_horizon,
            'top_features': top_features,
            'stacking_used': use_stacking and meta is not None,
            'lstm_available': lstm_pred_val is not None,
            'calibration': calibration_info,  # Tier 2.1
        }

    except Exception as e:
        logger.error(f"ML error: {traceback.format_exc()}")
        return {'error': str(e), 'models': []}


# ──────────────────────────────────────────────
# Multi-agent analysis
# ──────────────────────────────────────────────
def _run_agents(df, symbol, ml_pred, use_llm, llm_provider) -> dict:
    """Run the multi-agent system and return structured results."""
    try:
        comm_bus = AgentCommunicationBus()

        technical_agent = TechnicalAnalystAgent()
        sentiment_agent = SentimentAnalystAgent(use_llm=use_llm, llm_provider=llm_provider)
        risk_agent = RiskManagerAgent(use_garch=True)
        portfolio_manager = PortfolioManagerAgent(comm_bus, use_llm=use_llm, llm_provider=llm_provider)

        comm_bus.register_agent(technical_agent)
        comm_bus.register_agent(sentiment_agent)
        comm_bus.register_agent(risk_agent)
        comm_bus.register_agent(portfolio_manager)

        data = {
            'symbol': symbol,
            'price_data': df,
            'ml_prediction': ml_pred,
            'ml_result': ml_pred,  # full ml result dict for LLM context
            'portfolio': portfolio_manager.portfolio
        }

        pm_result = portfolio_manager.analyze(data)

        # Collect individual agent results from the PM's data
        individual = pm_result.data.get('individual_analyses', {})
        agents = {}
        for agent_key, info in individual.items():
            agents[agent_key] = {
                'recommendation': info.get('recommendation', 'N/A'),
                'confidence': round(float(info.get('confidence', 0)), 3),
                'reasoning': info.get('reasoning', ''),
            }

        return {
            'final_decision': pm_result.recommendation,
            'final_confidence': round(float(pm_result.confidence), 3),
            'final_reasoning': pm_result.reasoning,
            'individual': agents,
        }

    except Exception as e:
        logger.error(f"Agent error: {traceback.format_exc()}")
        return {
            'final_decision': 'ERROR',
            'final_confidence': 0,
            'final_reasoning': str(e),
            'individual': {},
        }


# ──────────────────────────────────────────────
# Hybrid score
# ──────────────────────────────────────────────
def _compute_hybrid_score(ml_result: dict, agent_result: dict,
                          ml_weight: float = 0.75) -> dict:
    """Compute the hybrid ML + LLM score.

    Direction-consistency rule: when ML and LLM disagree on direction,
    the recommendation is capped at MANTENER.  This prevents the UI from
    showing BUY while the projected price is going down (or vice-versa).

    Tier 3.3: ml_weight parametrizado para A/B sweep ML/LLM.
    Default 0.75 (mantenido). Valores razonables: 0.6-0.85.
    """
    ml_pred = ml_result.get('ensemble_prediction', 0)
    ml_conf = ml_result.get('ensemble_confidence', 0.5)

    # Convert ML return prediction to signal in [-1, 1].
    # Scale by 100 so ±1% return → ±1.0 signal (more sensitive than old *50).
    ml_signal = max(-1.0, min(1.0, float(ml_pred) * 100))

    # Convert agent decision to signal
    decision = agent_result.get('final_decision', 'MANTENER')
    decision_map = {
        'COMPRAR': 0.7, 'COMPRA': 0.7, 'COMPRA_FUERTE': 1.0,
        'VENDER': -0.7, 'VENTA': -0.7, 'VENTA_FUERTE': -1.0,
        'MANTENER': 0.0, 'ESPERAR': 0.0, 'HOLD': 0.0,
    }
    llm_signal = decision_map.get(decision, 0.0)
    llm_conf = agent_result.get('final_confidence', 0.5)

    # ML carries much more weight — it has a quantitative edge over LLM narrative.
    # Antes: 60/40.  Ahora: 75/25 (default) porque el LLM local (qwen2.5) aporta
    # sesgo ruidoso y bloqueaba señales ML buenas a través del veto de dirección.
    # Tier 3.3: ml_weight ahora parametrizable (default 0.75) para sweep A/B.
    ml_weight  = float(max(0.0, min(1.0, ml_weight)))
    llm_weight = 1.0 - ml_weight

    hybrid_score = float(ml_signal * ml_weight + llm_signal * llm_weight)

    # ── Direction penalty (ya NO veto) ──────────────────────────────────────
    # Cuando ML y LLM discrepan, penalizamos la confianza y reducimos la magnitud
    # del score un 40% — pero NO lo clampamos a MANTENER.  Si ML es muy decisivo,
    # su señal debe pasar.  (El veto anterior convertía la mayoría de los BUY
    # ML-correctos en MANTENER por disagreement con el LLM.)
    ml_dir  = 1 if ml_pred > 0.0005 else (-1 if ml_pred < -0.0005 else 0)
    llm_dir = 1 if llm_signal > 0 else (-1 if llm_signal < 0 else 0)
    direction_conflict = ml_dir != 0 and llm_dir != 0 and ml_dir != llm_dir
    if direction_conflict:
        hybrid_score *= 0.60

    hybrid_score = round(hybrid_score, 4)
    # Penalizamos la confianza híbrida si había conflicto de dirección
    hybrid_confidence_raw = float(ml_conf * ml_weight + llm_conf * llm_weight)
    if direction_conflict:
        hybrid_confidence_raw *= 0.80
    hybrid_confidence = round(hybrid_confidence_raw, 4)

    if hybrid_score > 0.4:
        recommendation = 'COMPRA_FUERTE'
    elif hybrid_score > 0.15:
        recommendation = 'COMPRA'
    elif hybrid_score < -0.4:
        recommendation = 'VENTA_FUERTE'
    elif hybrid_score < -0.15:
        recommendation = 'VENTA'
    else:
        recommendation = 'MANTENER'

    return {
        'score': hybrid_score,
        'confidence': hybrid_confidence,
        'recommendation': recommendation,
        'ml_signal': round(float(ml_signal), 4),
        'llm_signal': round(float(llm_signal), 4),
        'ml_weight': ml_weight,
        'llm_weight': llm_weight,
        'direction_conflict': bool(direction_conflict),
    }


# ──────────────────────────────────────────────
# API routes
# ──────────────────────────────────────────────
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})


@app.route('/api/ollama-status', methods=['GET'])
def ollama_status():
    """
    Health badge del LLM. Provider-aware:
      · ollama       → ping http://localhost:11434/api/tags
      · local        → reportamos siempre 'running' con TFG_LOCAL_LLM
      · huggingface  → 'running' con el modelo HF configurado
    Llamado por el frontend al cargar y cada 30 s.
    """
    provider = DEFAULT_LLM_PROVIDER

    # Local transformers (subproceso): siempre disponible mientras la app corra
    if provider == 'local':
        local_model = os.environ.get('TFG_LOCAL_LLM', 'Qwen/Qwen2.5-0.5B-Instruct')
        return jsonify({
            'status': 'running',
            'provider': 'local',
            'models': [local_model],
            'active_model': local_model,
            'model_count': 1,
            'message': f'LLM local ({local_model}) carga bajo demanda en subproceso.',
        })

    if provider == 'huggingface':
        hf_model = os.environ.get('TFG_HF_MODEL', 'mistralai/Mistral-7B-Instruct-v0.2')
        return jsonify({
            'status': 'running',
            'provider': 'huggingface',
            'models': [hf_model],
            'active_model': hf_model,
            'model_count': 1,
            'message': f'HuggingFace Inference API ({hf_model}).',
        })

    # provider == 'ollama' (default) → ping real
    import requests as req
    try:
        resp = req.get('http://localhost:11434/api/tags', timeout=3)
        if resp.status_code == 200:
            models = resp.json().get('models', [])
            model_names = [m['name'] for m in models]
            preferred_order = [
                'qwen2.5:7b', 'qwen2.5',
                'llama3.1:8b', 'llama3.1', 'llama3:8b', 'llama3',
                'mistral:7b', 'mistral',
                'deepseek-r1:7b', 'gemma2:9b', 'gemma2',
                'llama3.2', 'phi3', 'phi',
            ]
            active_model = model_names[0] if model_names else None
            for pref in preferred_order:
                matches = [m for m in model_names if m.startswith(pref.split(':')[0])]
                if matches:
                    active_model = matches[0]
                    break
            return jsonify({
                'status': 'running',
                'provider': 'ollama',
                'models': model_names,
                'active_model': active_model,
                'model_count': len(model_names),
            })
    except Exception:
        pass
    return jsonify({
        'status': 'offline',
        'provider': 'ollama',
        'models': [],
        'active_model': None,
        'message': 'Ollama no está corriendo. Ejecuta: ollama serve',
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    POST /api/analyze
    Body: { "symbol": "AAPL", "timeframe": "1y", "asset_type": "stock" }
    """
    body = request.get_json(force=True)
    symbol = body.get('symbol', '').strip()
    timeframe = body.get('timeframe', '1y')
    asset_type = body.get('asset_type', 'stock')

    if not symbol:
        return jsonify({'status': 'error', 'error': 'symbol is required'}), 400

    # Accept any timeframe — unknown ones default to 1 year
    if timeframe not in TIMEFRAME_MAP:
        timeframe = '1y'

    logger.info(f"=== Analysis requested: {symbol} ({asset_type}) timeframe={timeframe} ===")

    try:
        # SIEMPRE ejecutamos el análisis en un subproceso para aislar la
        # memoria del LLM. El subproceso elige modo full / lightweight según
        # TFG_LIGHTWEIGHT. Esto mantiene el server Flask en <100 MB y permite
        # cargar Qwen-1.5B (o mayor) sin riesgo de SIGKILL del Flask.
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        cmd = [sys.executable, '-u', os.path.join(here, 'cli_analyze.py'),
               symbol, timeframe, asset_type, DEFAULT_LLM_PROVIDER]
        env = os.environ.copy()
        env.setdefault('TFG_LLM_PROVIDER', DEFAULT_LLM_PROVIDER)
        # Timeout largo: pipeline completo con Optuna+CV+LLM puede ir a ~3-5 min
        timeout_s = 600 if not LIGHTWEIGHT_MODE else 300
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout_s, env=env, cwd=here,
            )
        except subprocess.TimeoutExpired:
            return jsonify({'status': 'error',
                            'error': f'subprocess timeout (>{timeout_s}s)'}), 504
        if proc.returncode != 0:
            logger.error(f"cli_analyze stderr:\n{proc.stderr[-2000:]}")
            return jsonify({
                'status': 'error',
                'error':  f'cli_analyze exit {proc.returncode}',
                'stderr': proc.stderr[-2000:],
            }), 500
        try:
            result = json.loads(proc.stdout)
        except Exception as je:
            return jsonify({
                'status': 'error', 'error': f'invalid json from cli: {je}',
                'stdout_tail': proc.stdout[-500:],
            }), 500
        return jsonify(result)
    except Exception as e:
        logger.error(f"Analysis error: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/suggestions', methods=['GET'])
def suggestions():
    """Returns popular stock and crypto symbols for autocomplete."""
    return jsonify({
        'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
                    'JPM', 'V', 'WMT', 'JNJ', 'UNH', 'NFLX', 'DIS', 'BA',
                    'AMD', 'INTC', 'CRM', 'PYPL', 'UBER'],
        'crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD',
                    'ADA-USD', 'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'MATIC-USD',
                    'LINK-USD', 'UNI-USD', 'LTC-USD', 'SHIB-USD', 'ATOM-USD'],
    })


# ──────────────────────────────────────────────
# Historical backtest endpoint
# ──────────────────────────────────────────────
HORIZON_MAP = {
    '1W': 7, '2W': 14, '1M': 30, '3M': 90, '6M': 180
}


@app.route('/api/historical-backtest', methods=['POST'])
def historical_backtest():
    """
    POST /api/historical-backtest
    Body: { symbol, historical_date, horizon, asset_type }

    Runs the FULL ML pipeline treating historical_date as "today" (no future data leakage).
    Then loads what actually happened in the following horizon_days and compares.
    """
    body         = request.get_json(force=True)
    symbol       = body.get('symbol', '').strip().upper()
    hist_date    = body.get('historical_date', '')
    horizon_key  = body.get('horizon', '1M')
    asset_type   = body.get('asset_type', 'stock')

    if not symbol:
        return jsonify({'status': 'error', 'error': 'symbol is required'}), 400
    if not hist_date:
        return jsonify({'status': 'error', 'error': 'historical_date is required'}), 400

    horizon_days = HORIZON_MAP.get(horizon_key, 30)

    try:
        result = _run_historical_backtest(symbol, hist_date, horizon_days, asset_type)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Backtest error: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


def _run_historical_backtest(symbol: str, historical_date: str,
                              horizon_days: int, asset_type: str) -> dict:
    """
    1. Load 2-year training window ending STRICTLY at historical_date (no leakage).
    2. Run the ML pipeline on this training data.
    3. Load actual data for the following horizon_days.
    4. Compare and return both paths + accuracy metrics.
    """
    import time as _time

    hist_dt     = datetime.strptime(historical_date, '%Y-%m-%d')
    train_start = (hist_dt - timedelta(days=730)).strftime('%Y-%m-%d')
    train_end   = historical_date  # strict cutoff — data_loader now enforces this

    logger.info(f"[BACKTEST] {symbol}: training window {train_start} → {train_end}")

    loader = FinancialDataLoader()
    df_train = loader.download_data(symbol, train_start, train_end, asset_type=asset_type)
    _time.sleep(0.5)  # avoid Yahoo rate limiting on rapid successive calls

    if df_train is None or df_train.empty or len(df_train) < 80:
        return {
            'status': 'error',
            'error': f'Insufficient historical training data for {symbol} before {historical_date}. Need at least 80 daily bars.'
        }

    # Clean + engineer features on training data
    processor = DataProcessor()
    df_train  = processor.clean_data(df_train)
    df_train  = processor.add_technical_indicators(df_train)

    # Run ML with the SAME prediction horizon as the evaluation window.
    # Previously, the model always predicted 5-day returns but was compared
    # against 30-day actual — a fundamental apples-vs-oranges error.
    ml_result = _train_and_predict_ml(df_train, asset_type=asset_type,
                                       prediction_horizon=horizon_days,
                                       symbol=symbol)

    if 'error' in ml_result:
        return {'status': 'error', 'error': f'ML training failed: {ml_result["error"]}'}

    entry_price   = float(df_train['Close'].iloc[-1])
    pred_return   = ml_result.get('ensemble_prediction', 0)
    pred_dir      = ml_result.get('ensemble_direction', 'LATERAL')
    pred_conf     = ml_result.get('ensemble_confidence', 0.5)
    target_price  = entry_price * (1 + pred_return)

    # ── Monte Carlo GBM path ──────────────────────────────────────────────
    # Drift = confidence-weighted blend of ML signal + historical momentum.
    # When ML confidence is low the historical trend (momentum) dominates so the
    # median path has a visible slope even when the ML predicts ~0% return.
    # This is a standard Bayesian prior approach, not faking results.
    hist_returns     = df_train['Close'].pct_change().dropna()
    daily_vol        = float(hist_returns.std())            # realized daily σ
    ml_daily_drift   = pred_return / max(horizon_days, 1)  # ML signal per day
    hist_daily_drift = float(hist_returns.mean())           # historical momentum
    # Cap historical drift: ±0.3 %/day prevents runaway extrapolation
    hist_daily_drift = float(np.clip(hist_daily_drift, -0.003, 0.003))
    # Blend: 42% confidence → 42% ML + 58% historical trend
    blend_w  = float(pred_conf)  # in [0.10, 0.85]
    daily_drift = blend_w * ml_daily_drift + (1.0 - blend_w) * hist_daily_drift

    rng  = np.random.default_rng(seed=42)  # reproducible
    n_sims = 2000  # more paths → smoother percentile bands
    sim_paths = np.empty((n_sims, horizon_days + 1))
    sim_paths[:, 0] = entry_price
    for t in range(1, horizon_days + 1):
        shocks = rng.standard_normal(n_sims)
        sim_paths[:, t] = sim_paths[:, t - 1] * (1.0 + daily_drift + daily_vol * shocks)
    sim_paths = np.maximum(sim_paths, 0.0)  # prices can't go negative

    median_path = np.median(sim_paths, axis=0)
    ci_lower    = np.percentile(sim_paths, 10, axis=0)  # 10th percentile (wider band)
    ci_upper    = np.percentile(sim_paths, 90, axis=0)  # 90th percentile (wider band)

    predicted_path = [round(float(v), 4) for v in median_path]
    ci_lower_path  = [round(float(v), 4) for v in ci_lower]
    ci_upper_path  = [round(float(v), 4) for v in ci_upper]

    # Load ACTUAL data after historical_date
    actual_start = historical_date
    actual_end   = (hist_dt + timedelta(days=horizon_days + 15)).strftime('%Y-%m-%d')

    logger.info(f"[BACKTEST] Loading actual data: {actual_start} → {actual_end}")
    df_actual = loader.download_data(symbol, actual_start, actual_end, asset_type=asset_type)

    if df_actual is None or df_actual.empty:
        return {
            'status': 'error',
            'error': f'No actual data available after {historical_date} for {symbol}. '
                     'The date may be too recent or the symbol unavailable for this period.'
        }

    # Trim to exactly horizon_days trading days (or fewer if not available)
    df_actual = df_actual.iloc[:horizon_days]
    if df_actual.empty:
        return {'status': 'error', 'error': 'No post-date data available for comparison'}

    # Build actual price path starting from entry_price
    actual_prices = [entry_price] + [round(float(p), 4) for p in df_actual['Close'].tolist()]
    actual_dates  = [historical_date] + [str(d.date()) for d in df_actual.index]

    exit_price      = float(df_actual['Close'].iloc[-1])
    actual_return   = (exit_price - entry_price) / entry_price
    actual_dir      = 'SUBE' if actual_return > 0.001 else 'BAJA' if actual_return < -0.001 else 'LATERAL'
    actual_ret_pct  = round(actual_return * 100, 4)
    pred_ret_pct    = round(pred_return * 100, 4)

    # Accuracy
    direction_correct  = (pred_dir == actual_dir)
    return_error_pct   = round(abs(pred_ret_pct - actual_ret_pct), 4)
    if return_error_pct < 2:
        rating = 'EXCELLENT'
    elif return_error_pct < 5:
        rating = 'GOOD'
    else:
        rating = 'POOR'

    # Align predicted/CI paths to same length as actual dates
    n = len(actual_dates)
    pred_aligned      = predicted_path[:n]
    ci_lower_aligned  = ci_lower_path[:n]
    ci_upper_aligned  = ci_upper_path[:n]

    return {
        'status': 'success',
        'symbol': symbol,
        'historical_date': historical_date,
        'horizon_days': horizon_days,

        'prediction': {
            'entry_price':          round(entry_price, 4),
            'predicted_return_pct': pred_ret_pct,
            'predicted_direction':  pred_dir,
            'target_price':         round(target_price, 4),
            'confidence':           pred_conf,
            'ml_models':            ml_result.get('models', []),
            'daily_vol_pct':        round(daily_vol * 100, 3),
        },

        'actual': {
            'actual_return_pct': actual_ret_pct,
            'actual_direction':  actual_dir,
            'exit_price':        round(exit_price, 4),
        },

        'accuracy': {
            'direction_correct':  direction_correct,
            'return_error_pct':   return_error_pct,
            'return_error_rating': rating,
        },

        'chart': {
            'dates':          actual_dates,
            'predicted_path': pred_aligned,
            'ci_lower':       ci_lower_aligned,
            'ci_upper':       ci_upper_aligned,
            'actual_path':    actual_prices[:len(actual_dates)],
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRADING BOT ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

from trading_bot.bot_engine import AutonomousTradingBot, BotConfig
from models.ml_models.neural_networks_sklearn import NeuralEnsemble


def _generate_bot_predictions(df: pd.DataFrame, asset_type: str = 'stock',
                               train_ratio: float = 0.70) -> dict:
    """
    Entrena el ensemble completo (RF+XGB+LGB+MLP+RBF) en el periodo de entrenamiento
    y genera predicciones para el periodo de test (walk-forward).

    Returns dict con:
      predictions  : array de retornos predichos para el test set
      confidences  : array de confianzas [0,1]
      test_start_idx: indice en df donde empieza el test set
      model_metrics : metricas de CV de cada modelo
    """
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                                  RandomForestRegressor, HistGradientBoostingRegressor)
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
    from sklearn.linear_model import LogisticRegression

    prediction_horizon = 5 if asset_type != 'crypto' else 7

    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.create_features(df, prediction_horizon=prediction_horizon)
    feature_cols = feature_engineer.feature_names

    if len(df_features) < 80:
        raise ValueError("Datos insuficientes para entrenamiento del bot (minimo 80 muestras).")

    # Usar clasificación de dirección (0=baja, 1=sube) en lugar de regresión de magnitud.
    # La magnitud del retorno es muy difícil de predecir; la dirección es más estable
    # y permite calibrar confianza como probabilidad (predict_proba).
    # seq_length=1: usamos solo la fila más reciente en lugar de apilar 5 días.
    # Motivo (Red Team): con 27 features, seq=5 genera 135 columnas de las cuales
    # muchas son redundantes (sma_50, return_10d ya incorporan lookback), lo que
    # amplifica overfitting cuando el clasificador tiene que discriminar en un
    # régimen OOS distinto al de entrenamiento. Con seq=1 el modelo ve
    # directamente el estado ACTUAL del mercado sin hinchazón dimensional.
    X, y_dir = feature_engineer.prepare_ml_data(df_features, feature_cols, 'target_direction', seq_length=1)
    y = y_dir  # usamos dirección (0/1) para entrenar los clasificadores

    # ── Split train / test ANTES de feature selection (anti-leakage) ─────────
    n_total = len(X)
    n_train = int(n_total * train_ratio)
    # Los últimos prediction_horizon filas del train tienen targets que usan
    # precios del test → los excluimos del entrenamiento (Red Team Finding 1)
    n_safe_train = max(30, n_train - prediction_horizon)

    X_train_raw = X[:n_safe_train]
    y_train     = y[:n_safe_train]
    X_test      = X[n_train:]
    y_test      = y[n_train:]

    if len(X_test) < 10:
        raise ValueError("Periodo de test demasiado corto (< 10 muestras).")

    # ── Feature selection SOLO sobre train (anti-leakage, Red Team Finding 2) ─
    # Usar clasificador para feature selection (coherente con y_dir = 0/1)
    n_features_total = X_train_raw.shape[1]
    selected_indices = np.arange(n_features_total)
    if n_features_total > 40 and n_safe_train >= 100:
        try:
            quick_xgb_fs = xgb.XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.6,
                reg_alpha=0.5, reg_lambda=1.0,
                eval_metric='logloss',
                random_state=42, n_jobs=-1
            )
            quick_xgb_fs.fit(X_train_raw, y_train.astype(int))
            importances = quick_xgb_fs.feature_importances_
            n_keep = min(30, max(15, n_features_total // 3))
            selected_indices = np.argsort(importances)[-n_keep:]
        except Exception:
            selected_indices = np.arange(n_features_total)

    X_train = X_train_raw[:, selected_indices]
    X_test  = X_test[:, selected_indices]

    # Hiperparametros segun tipo de activo (clasificadores)
    if asset_type == 'crypto':
        rf_p   = dict(n_estimators=150, max_depth=5, random_state=42, n_jobs=-1)
        xgb_p  = dict(n_estimators=150, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.7,
                      eval_metric='logloss',
                      random_state=42, n_jobs=-1)
        lgb_p  = dict(n_estimators=150, max_depth=4, learning_rate=0.05,
                      min_child_samples=15, random_state=42, verbose=-1)
        hgb_p  = dict(max_iter=150, max_depth=4, learning_rate=0.05, random_state=42)
    else:
        rf_p   = dict(n_estimators=200, max_depth=7, random_state=42, n_jobs=-1)
        xgb_p  = dict(n_estimators=200, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.7,
                      eval_metric='logloss',
                      random_state=42, n_jobs=-1)
        lgb_p  = dict(n_estimators=200, max_depth=5, learning_rate=0.05,
                      min_child_samples=15, random_state=42, verbose=-1)
        hgb_p  = dict(max_iter=200, max_depth=5, learning_rate=0.05, random_state=42)

    y_train_int = y_train.astype(int)
    y_test_int  = y_test.astype(int)

    # ── WALK-FORWARD RETRAINING ─────────────────────────────────────────────
    # En lugar de entrenar UNA vez con el 70% inicial y predecir todo el test
    # (expuesto a régimen-shift), reentrenamos el ensemble cada `retrain_freq`
    # días del test usando ventana expanding (todo el histórico disponible
    # hasta `prediction_horizon` días antes del bloque, para purgar solapes).
    # Cada bloque se predice con un ensemble FRESCO cuyos pesos se estiman vía
    # TimeSeriesSplit sobre su propio histórico (sin leakage del bloque de
    # predicción).
    from sklearn.model_selection import TimeSeriesSplit

    retrain_freq = 20  # reentrenar cada 20 días (~1 mes de trading)
    n_test = len(X_test)
    model_names = ['random_forest', 'xgboost', 'lightgbm', 'hist_gradient', 'logistic']

    # Probas concatenadas por modelo y del ensemble
    per_model_proba = {name: np.zeros(n_test) for name in model_names}
    ensemble_proba  = np.zeros(n_test)
    cv_acc_per_block = {name: [] for name in model_names}

    def _build_model(name):
        if name == 'random_forest':  return RandomForestClassifier(**rf_p)
        if name == 'xgboost':        return xgb.XGBClassifier(**xgb_p)
        if name == 'lightgbm':       return lgb.LGBMClassifier(**lgb_p)
        if name == 'hist_gradient':  return HistGradientBoostingClassifier(**hgb_p)
        if name == 'logistic':       return LogisticRegression(C=0.1, max_iter=500, random_state=42)
        raise ValueError(name)

    def _fit_predict(name, X_tr, y_tr, X_pr):
        """Entrena `name` sobre (X_tr, y_tr) y devuelve P(sube) para X_pr."""
        model = _build_model(name)
        if name == 'logistic':
            sc = StandardScaler().fit(X_tr)
            model.fit(sc.transform(X_tr), y_tr)
            return model.predict_proba(sc.transform(X_pr))[:, 1]
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_pr)[:, 1]

    for block_start in range(0, n_test, retrain_freq):
        block_end = min(block_start + retrain_freq, n_test)
        # El training "walk-forward" crece con cada bloque: todo lo que está
        # antes del bloque menos la zona de purga (horizonte de predicción).
        # Indices en el array X global: train = [0 .. n_train + block_start - horizon]
        safe_end = n_train + block_start - prediction_horizon
        if safe_end < 50:
            continue
        X_tr = X[:safe_end, selected_indices]
        y_tr = y[:safe_end].astype(int)
        X_pr = X_test[block_start:block_end]

        # CV sobre TRAIN de este bloque — estima edge por modelo sin tocar el bloque
        block_cv_acc = {name: 0.5 for name in model_names}
        try:
            tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(X_tr) // 60)))
            cv_accs_local = {name: [] for name in model_names}
            for tr_idx, va_idx in tscv.split(X_tr):
                if len(tr_idx) < 30 or len(va_idx) < 5:
                    continue
                yt = y_tr[tr_idx]; yv = y_tr[va_idx]
                for name in model_names:
                    p = _fit_predict(name, X_tr[tr_idx], yt, X_tr[va_idx])
                    cv_accs_local[name].append(float(accuracy_score(yv, (p >= 0.5).astype(int))))
            block_cv_acc = {n: (sum(v)/len(v) if v else 0.5) for n, v in cv_accs_local.items()}
        except Exception as _cv_err:
            logger.debug(f"CV bloque {block_start} falló: {_cv_err}")

        for name in model_names:
            cv_acc_per_block[name].append(block_cv_acc[name])

        # Pesos del bloque según edge (acc − 0.50)² + ε
        b_weights = {n: (max(block_cv_acc[n] - 0.50, 0.0) ** 2 + 1e-4) for n in model_names}
        total_bw = sum(b_weights.values())

        # Entrenar cada modelo sobre TODO el train y predecir el bloque completo
        block_ensemble = np.zeros(block_end - block_start)
        for name in model_names:
            p = _fit_predict(name, X_tr, y_tr, X_pr)
            per_model_proba[name][block_start:block_end] = p
            block_ensemble += p * (b_weights[name] / total_bw)
        ensemble_proba[block_start:block_end] = block_ensemble

    # Métricas OOS por modelo (accuracy real sobre el test set completo)
    model_metrics = []
    for name in model_names:
        p_full = per_model_proba[name]
        pred_dir_m = (p_full >= 0.5).astype(int)
        acc_m = float(accuracy_score(y_test_int, pred_dir_m))
        model_metrics.append({
            'name': name,
            'accuracy': round(acc_m, 4),
            'rmse': round(float(np.sqrt(mean_squared_error(y_test_int, p_full))), 6),
            'r2':   round(float(r2_score(y_test_int, p_full)), 6),
            'cv_accuracy_mean': round(
                (sum(cv_acc_per_block[name]) / len(cv_acc_per_block[name]))
                if cv_acc_per_block[name] else 0.5, 4
            ),
            'n_retrain_blocks': len(cv_acc_per_block[name]),
        })

    # Convertir probabilidad → señal de retorno predicho
    # La magnitud es ADAPTATIVA: usamos la volatilidad histórica esperada sobre el
    # horizonte de predicción como tamaño "natural" del movimiento esperado. Antes
    # usábamos 0.01 fijo lo que hacía que Kelly no escalara por activo (un activo
    # volátil y uno estable generaban el mismo sizing).
    try:
        _daily_ret = df['Close'].pct_change().dropna()
        _daily_vol_est = float(_daily_ret.std())
    except Exception:
        _daily_vol_est = 0.015
    # Movimiento esperado en `prediction_horizon` días ≈ σ_daily · √h
    horizon_move_est = _daily_vol_est * np.sqrt(prediction_horizon)
    # Magnitud calibrada: entre 0.5% y 4%. En cripto (vol alta) será ~3-4%, en stocks ~0.8-1.8%.
    PRED_MAGNITUDE = float(np.clip(horizon_move_est * 0.60, 0.005, 0.04))
    # proba=0.50 → pred=0.  proba=0.80 → pred = 0.6 · PRED_MAGNITUDE (no saturado)
    # proba=1.00 → pred = PRED_MAGNITUDE (máximo)
    edge = (ensemble_proba - 0.5) * 2  # [-1, +1]
    ensemble_preds = edge * PRED_MAGNITUDE

    # Confianza = distancia desde 0.5 (azar) normalizada a [0,1]
    # P=0.5 → conf=0.0 (sin edge), P=1.0 → conf=1.0 (certeza total).
    # El rango se amplía a [0.15, 0.90] para que señales muy claras puedan superar
    # el umbral Kelly (antes el cap de 0.85 limitaba demasiado con b_min=0.5).
    raw_confidence = np.abs(ensemble_proba - 0.5) * 2
    confidences = np.clip(0.15 + raw_confidence * 0.78, 0.15, 0.90)

    # ── Accuracy gate: solo operar si el clasificador supera al azar ────────
    # Un clasificador con accuracy ≤ 50% tiene edge negativo → no operar.
    # y_test_int = etiquetas 0/1 de dirección real; ensemble_preds con signo.
    if len(y_test_int) >= 10:
        pred_dir_gate = (ensemble_preds > 0).astype(int)
        direction_accuracy = float(np.mean(pred_dir_gate == y_test_int))
        logger.info(f"Direction accuracy OOS: {direction_accuracy:.3f} ({len(y_test_int)} samples)")
        # Gate estricto: sólo `< 0.50` bloquea. 0.500 exacto es el empate con el
        # azar; no es edge negativo. Bloquearlo eliminaba señales buenas por
        # variancia en periodos cortos (p.ej. 82/164 aciertos). Mantenemos el
        # veto sólo cuando la accuracy está *claramente* por debajo del azar.
        if direction_accuracy < 0.50:
            logger.warning(
                f"Direction accuracy {direction_accuracy:.3f} < 0.50 — modelo sin edge. "
                "Bot no operará en este activo (señales zeroed)."
            )
            ensemble_preds = np.zeros_like(ensemble_preds)
            confidences = np.zeros_like(confidences)
    else:
        direction_accuracy = 0.0

    # Indice en df original donde empieza el test
    # df_features.iloc[n_train] corresponde al n_train-esimo punto tras el lag
    test_start_idx_in_df = df_features.index[n_train]
    try:
        test_start_loc = df.index.get_loc(test_start_idx_in_df)
    except Exception:
        test_start_loc = n_train  # fallback

    return {
        'predictions':         ensemble_preds,
        'confidences':         confidences,
        'test_start_idx':      test_start_loc,
        'model_metrics':       model_metrics,
        'n_train':             n_train,
        'n_test':              len(X_test),
        'feature_cols':        feature_cols,
        'selected_indices':    selected_indices.tolist(),
        'prediction_horizon':  prediction_horizon,
        'direction_accuracy':  direction_accuracy,
    }


@app.route('/api/bot/backtest', methods=['POST'])
def bot_backtest():
    """
    POST /api/bot/backtest
    Body: {
      symbol, asset_type,
      initial_capital (opt, default 100000),
      signal_percentile (opt, default 0.75),
      max_position_pct (opt, default 0.30),
      allow_short (opt, default true),
      years (opt, default 2)
    }

    Entrena el sistema ML completo (RF+XGB+LGB+MLP+RBF) y simula el bot
    de trading autonomo en el periodo de test out-of-sample.
    Devuelve equity curve, trades, metricas monetarias y comparativa con
    buy & hold.
    """
    body = request.get_json(force=True)
    symbol       = body.get('symbol', '').strip().upper()
    asset_type   = body.get('asset_type', 'stock')
    init_capital = float(body.get('initial_capital', 100_000))
    sig_pct      = float(body.get('signal_percentile', 0.75))
    max_pos      = float(body.get('max_position_pct', 0.30))
    allow_short  = bool(body.get('allow_short', True))
    years        = int(body.get('years', 2))

    if not symbol:
        return jsonify({'status': 'error', 'error': 'symbol is required'}), 400

    logger.info(f"=== Bot backtest: {symbol} ({asset_type}), {years}y, capital=${init_capital:,.0f} ===")

    try:
        # 1. Descargar datos
        end_date   = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')

        loader = FinancialDataLoader()
        df = loader.download_data(symbol, start_date, end_date, asset_type=asset_type)

        if df is None or df.empty or len(df) < 100:
            return jsonify({'status': 'error',
                            'error': f'Datos insuficientes para {symbol}. Se necesitan al menos 100 dias.'}), 400

        processor = DataProcessor()
        df = processor.clean_data(df)
        df = processor.add_technical_indicators(df)

        # 2. Generar predicciones walk-forward
        bot_preds = _generate_bot_predictions(df, asset_type=asset_type)

        # 3. Ejecutar simulacion del bot
        cfg = BotConfig(
            initial_capital=init_capital,
            signal_percentile=sig_pct,
            max_position_pct=max_pos,
            allow_short=allow_short,
        )
        bot = AutonomousTradingBot(config=cfg)
        result = bot.run_backtest(
            price_df=df,
            predictions=bot_preds['predictions'],
            confidences=bot_preds['confidences'],
            test_start_idx=bot_preds['test_start_idx'],
            symbol=symbol,
        )

        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'asset_type': asset_type,
            'config': {
                'initial_capital': init_capital,
                'signal_percentile': sig_pct,
                'max_position_pct': max_pos,
                'allow_short': allow_short,
                'years_history': years,
            },
            'model_metrics': bot_preds['model_metrics'],
            'training_info': {
                'n_train': bot_preds['n_train'],
                'n_test':  bot_preds['n_test'],
                'prediction_horizon_days': bot_preds['prediction_horizon'],
            },
            **result,
        })

    except Exception as e:
        logger.error(f"Bot backtest error: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/bot/signal', methods=['POST'])
def bot_signal():
    """
    POST /api/bot/signal
    Body: { symbol, asset_type }

    Genera la señal de trading actual para un simbolo usando el ensemble completo.
    Devuelve accion recomendada, sizing sugerido, y detalle del modelo.
    """
    body       = request.get_json(force=True)
    symbol     = body.get('symbol', '').strip().upper()
    asset_type = body.get('asset_type', 'stock')

    if not symbol:
        return jsonify({'status': 'error', 'error': 'symbol is required'}), 400

    try:
        result = run_full_analysis(symbol=symbol, timeframe='1y',
                                   asset_type=asset_type, use_llm=False)

        if result.get('status') != 'success':
            return jsonify(result)

        ml = result.get('ml', {})
        pred_return = float(ml.get('ensemble_prediction', 0))
        confidence  = float(ml.get('ensemble_confidence', 0.5))
        signal_strength = abs(pred_return) * confidence

        # Determinar accion
        if signal_strength < 0.001:
            action = 'HOLD'
            reason = 'Señal demasiado debil para operar'
        elif pred_return > 0:
            action = 'LONG'
            reason = f'Prediccion alcista ({pred_return*100:+.2f}%), confianza {confidence:.0%}'
        else:
            action = 'SHORT'
            reason = f'Prediccion bajista ({pred_return*100:+.2f}%), confianza {confidence:.0%}'

        # Sizing sugerido (Kelly fraccionado)
        p = min(max(confidence, 0.1), 0.9)
        b = max(abs(pred_return) / 0.01, 0.5)
        kelly = max(0, (p * b - (1 - p)) / b)
        suggested_size_pct = min(kelly * 0.25 * 100, 30)  # max 30%

        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'signal': {
                'action':           action,
                'predicted_return': round(pred_return * 100, 3),
                'confidence':       round(confidence, 3),
                'signal_strength':  round(signal_strength, 5),
                'suggested_size_pct': round(suggested_size_pct, 1),
                'reason':           reason,
            },
            'current_price':    result['data_info'].get('current_price'),
            'volatility':       result['data_info'].get('annualized_volatility'),
            'ml_models':        ml.get('models', []),
            'hybrid_score':     result.get('hybrid', {}).get('score'),
        })

    except Exception as e:
        logger.error(f"Bot signal error: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# BOT HISTORICAL BACKTEST — sin fuga de datos
# ══════════════════════════════════════════════════════════════════════════════

def _generate_bot_predictions_by_date(
    df: pd.DataFrame,
    asset_type: str,
    test_start_date: str,
) -> dict:
    """
    Igual que _generate_bot_predictions pero el split train/test se hace
    por FECHA, no por ratio.

    Garantía anti-leakage:
      - df solo contiene datos hasta la fecha fin del backtest (caller lo trunca).
      - El modelo se entrena EXCLUSIVAMENTE en datos anteriores a test_start_date.
      - Las predicciones del período de test nunca influyen en el entrenamiento.

    Args:
        df             : DataFrame de precios con indicadores.  Solo tiene datos
                         hasta end_date (sin futuros).
        asset_type     : 'stock' o 'crypto'.
        test_start_date: Primer día del período de test (formato 'YYYY-MM-DD').
                         Todo lo anterior se usa para entrenar.
    """
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    test_start_dt = pd.Timestamp(test_start_date)
    prediction_horizon = 5 if asset_type != 'crypto' else 7

    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.create_features(df, prediction_horizon=prediction_horizon)
    feature_cols = feature_engineer.feature_names

    if len(df_features) < 80:
        raise ValueError("Datos insuficientes para el backtest (mínimo 80 muestras).")

    X_all, y_all = feature_engineer.prepare_ml_data(df_features, feature_cols, 'target_return')
    dates = df_features.index

    # ── Split estrictamente por fecha (NO por ratio) ─────────────────────────
    train_mask = dates < test_start_dt
    test_mask  = dates >= test_start_dt
    n_train    = int(train_mask.sum())
    n_test     = int(test_mask.sum())

    if n_train < 50:
        raise ValueError(
            f"Período de entrenamiento demasiado corto ({n_train} días). "
            "Necesitas al menos 50 días de historia antes de la fecha de inicio del test."
        )
    if n_test < 5:
        raise ValueError(
            f"Período de test demasiado corto ({n_test} días). "
            "Amplía el rango de fechas del backtest."
        )

    # Los últimos prediction_horizon filas del train tienen targets que usan
    # precios del test → los excluimos para evitar leakage (Red Team Finding 1)
    n_safe_train = max(30, n_train - prediction_horizon)

    X_train_raw = X_all[:n_safe_train]
    y_train     = y_all[:n_safe_train]
    X_test      = X_all[n_train:]
    y_test      = y_all[n_train:]

    # ── Feature selection (solo sobre train_safe, sin ver test) ─────────────
    selected_indices = np.arange(X_train_raw.shape[1])
    if X_train_raw.shape[1] > 60 and n_safe_train >= 100:
        try:
            quick_xgb = xgb.XGBRegressor(
                n_estimators=80, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.7, random_state=42, n_jobs=-1,
            )
            quick_xgb.fit(X_train_raw, y_train)
            importances = quick_xgb.feature_importances_
            n_keep = min(50, max(20, X_train_raw.shape[1] // 2))
            selected_indices = np.argsort(importances)[-n_keep:]
        except Exception:
            pass

    X_train = X_train_raw[:, selected_indices]
    X_test  = X_test[:, selected_indices]

    # ── Calidad pre-test del activo (sin leakage) ────────────────────────────
    # Walk-forward CV sobre el TRAINING set: nunca toca el test.
    # Usamos un xgb rápido como proxy del ensemble. R² promedio entre folds
    # es nuestra estimación de "qué tan modelable es este activo".
    # Activos con val_r2 << 0 → el ML no aporta señal real → mejor descartar.
    train_validation_r2 = 0.0
    try:
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3, gap=prediction_horizon)
        fold_r2s = []
        for tr_idx, va_idx in tscv.split(X_train):
            if len(tr_idx) < 50 or len(va_idx) < 10:
                continue
            proxy = xgb.XGBRegressor(
                n_estimators=80, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.7,
                random_state=42, n_jobs=-1,
            )
            proxy.fit(X_train[tr_idx], y_train[tr_idx])
            pv = proxy.predict(X_train[va_idx])
            fold_r2s.append(float(r2_score(y_train[va_idx], pv)))
        if fold_r2s:
            train_validation_r2 = float(np.mean(fold_r2s))
    except Exception as _cv_err:
        logger.debug(f"train_validation_r2 cv error: {_cv_err}")

    # ── Hiperparámetros según tipo de activo ─────────────────────────────────
    if asset_type == 'crypto':
        rf_p  = dict(n_estimators=150, max_depth=5, random_state=42, n_jobs=-1)
        xgb_p = dict(n_estimators=150, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.7, random_state=42, n_jobs=-1)
        lgb_p = dict(n_estimators=150, max_depth=4, learning_rate=0.05,
                     min_child_samples=15, random_state=42, verbose=-1)
        hgb_p = dict(max_iter=150, max_depth=4, learning_rate=0.05, random_state=42)
    else:
        rf_p  = dict(n_estimators=200, max_depth=7, random_state=42, n_jobs=-1)
        xgb_p = dict(n_estimators=200, max_depth=5, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.7, random_state=42, n_jobs=-1)
        lgb_p = dict(n_estimators=200, max_depth=5, learning_rate=0.05,
                     min_child_samples=15, random_state=42, verbose=-1)
        hgb_p = dict(max_iter=200, max_depth=5, learning_rate=0.05, random_state=42)

    # ── Entrenar modelos base (solo en train, sin leakage) ───────────────────
    rf    = RandomForestRegressor(**rf_p).fit(X_train, y_train)
    xgb_m = xgb.XGBRegressor(**xgb_p).fit(X_train, y_train)
    lgb_m = lgb.LGBMRegressor(**lgb_p).fit(X_train, y_train)
    hgb_m = HistGradientBoostingRegressor(**hgb_p).fit(X_train, y_train)

    # ElasticNet (lineal) — verdadera diversidad vs tree-based
    _scaler2  = StandardScaler()
    X_tr_sc2  = _scaler2.fit_transform(X_train)
    X_te_sc2  = _scaler2.transform(X_test)
    en_m2     = ElasticNet(alpha=0.005, l1_ratio=0.5, max_iter=3000, random_state=42)
    en_m2.fit(X_tr_sc2, y_train)

    model_metrics = []
    base_preds_test = {}
    for name, model, X_te in [('random_forest', rf, X_test),
                               ('xgboost', xgb_m, X_test),
                               ('lightgbm', lgb_m, X_test),
                               ('hist_gradient', hgb_m, X_test),
                               ('elasticnet', en_m2, X_te_sc2)]:
        p = model.predict(X_te)
        if name == 'elasticnet':
            p = np.clip(p, -0.15, 0.15)
        base_preds_test[name] = p
        model_metrics.append({
            'name': name,
            'rmse': round(float(np.sqrt(mean_squared_error(y_test, p))), 6),
            'mae':  round(float(mean_absolute_error(y_test, p)), 6),
            'r2':   round(float(r2_score(y_test, p)), 6),
        })

    # ── Redes neuronales (MLP + RBF) ─────────────────────────────────────────
    try:
        neural = NeuralEnsemble()
        mlp_metrics, rbf_metrics = neural.fit_evaluate(X_train, y_train, n_splits=3)
        neural_preds_test = neural.predict(X_test)
        mlp_preds_test, rbf_preds_test = neural.predict_individual(X_test)
        model_metrics.append({**mlp_metrics, 'name': 'mlp_neural'})
        model_metrics.append({**rbf_metrics, 'name': 'rbf_neural'})
        base_preds_test['mlp'] = mlp_preds_test
        base_preds_test['rbf'] = rbf_preds_test
    except Exception as nn_err:
        logger.warning(f"Neural ensemble error en backtest histórico: {nn_err}")

    # ── Ensemble ponderado por inverse-RMSE, pero penalizando R² negativo ────
    # Un modelo con R² < 0 predice peor que la media y NO debe tener peso sustancial.
    # Usamos un peso combinado: inverse-RMSE × max(R²+0.05, 0.02)² que colapsa a casi
    # cero cuando el R² es muy negativo.
    n_preds = len(X_test)
    rmse_map = {m['name']: m['rmse'] for m in model_metrics}
    r2_map   = {m['name']: m['r2']   for m in model_metrics}
    weights = {}
    for name in base_preds_test:
        rmse_w = 1.0 / max(rmse_map.get(name, 0.01), 1e-9)
        r2_adj = max(r2_map.get(name, -0.5) + 0.05, 0.02)  # R²=-0.3 → 0.02, R²=0 → 0.05, R²=0.3 → 0.35
        weights[name] = rmse_w * (r2_adj ** 2)
    total_w = sum(weights.values())
    ensemble_preds = np.zeros(n_preds)
    for name, preds in base_preds_test.items():
        p_trimmed = np.asarray(preds[:n_preds])
        ensemble_preds += p_trimmed * (weights[name] / total_w)

    # ── Confianza basada en acuerdo de dirección entre modelos ───────────────
    all_preds_matrix = np.column_stack(
        [np.asarray(p[:n_preds]) for p in base_preds_test.values()]
    )
    ens_sign = np.sign(ensemble_preds).reshape(-1, 1)
    model_signs = np.sign(all_preds_matrix)
    agreement = np.mean(model_signs == ens_sign, axis=1)
    confidences = np.clip(0.25 + agreement * 0.60, 0.15, 0.90)
    # Capear por magnitud de la predicción: señales microscópicas → confianza baja
    pred_mag = np.abs(ensemble_preds)
    conf_cap = np.clip(0.35 + pred_mag / 0.002 * 0.55, 0.30, 0.90)
    confidences = np.minimum(confidences, conf_cap)
    # Penalizar por calidad media del ensemble (si los modelos son malos en OOS,
    # la confianza no puede ser alta sea cual sea el acuerdo)
    mean_r2 = float(np.mean(list(r2_map.values()))) if r2_map else 0.0
    if mean_r2 < -0.10:
        confidences *= 0.70  # claro underperformer → rebajar confianza un 30%
    elif mean_r2 < 0.0:
        confidences *= 0.85

    # ── Índice en df original del primer día de test ──────────────────────────
    test_start_idx_in_df = df_features.index[n_train]
    try:
        test_start_loc = df.index.get_loc(test_start_idx_in_df)
    except Exception:
        test_start_loc = n_train

    return {
        'predictions':      ensemble_preds,
        'confidences':      confidences,
        'test_start_idx':   test_start_loc,
        'model_metrics':    model_metrics,
        'mean_r2':          float(mean_r2),
        'train_validation_r2': float(train_validation_r2),
        'n_train':          n_train,
        'n_test':           n_test,
        'prediction_horizon': prediction_horizon,
        'selected_indices': selected_indices.tolist() if hasattr(selected_indices, 'tolist') else list(selected_indices),
        'train_end_date':   str(dates[n_train - 1].date()) if n_train > 0 else '',
        'test_start_date':  str(dates[n_train].date()) if n_train < len(dates) else '',
        'test_end_date':    str(dates[-1].date()),
    }


@app.route('/api/paper/autonomous-backtest', methods=['POST'])
def paper_autonomous_backtest():
    """
    POST /api/paper/autonomous-backtest
    Simula el bot autónomo escaneando TODOS los activos de la watchlist cada día
    y ejecutando la mejor señal — idéntico comportamiento al Paper Trading Live.

    Body: {
      start_date      : "YYYY-MM-DD"
      end_date        : "YYYY-MM-DD"
      initial_capital : (opt, default 100000)
      allow_short     : (opt, default true)
      signal_percentile: (opt, default 0.75)
      kelly_scale      : (opt, default 0.25)
    }
    """
    body        = request.get_json(force=True) or {}
    start_date  = body.get('start_date', '')
    end_date    = body.get('end_date', '')
    init_cap      = float(body.get('initial_capital', 100_000))
    allow_short   = bool(body.get('allow_short', True))
    sig_pct       = float(body.get('signal_percentile', 0.80))
    kelly_sc      = float(body.get('kelly_scale', 0.30))
    commission_rate = float(body.get('commission_rate', 0.001))  # default 0.1%
    # mode = 'trend' (default) → trend-follower puro Faber/Antonacci (sin ML, señal SMA50/SMA200)
    # mode = 'ml'              → ensemble + R² gate (V5)
    #
    # Veredicto del benchmark multi-ventana (compare_modes.py, 6×2y aleatorias 2018-24):
    #   · ML    avg ret  -2.66%  ·  Sharpe 0.01  ·  PF 0.46  ·  0% ventanas positivas
    #   · TREND avg ret +23.36%  ·  Sharpe 0.51  ·  PF 1.46  ·  100% ventanas positivas
    # → TREND gana en todas las métricas. El ML no genera alpha en este universo.
    mode          = str(body.get('mode', 'trend')).lower()

    # ── Flags operativos (defaults dependen del mode) ────────────────────────
    # Defaults para mode='trend' = variante MED del benchmark (mejor risk-adj):
    #   · trend_reverse_exit=True  (salir en death cross — natural en trend mode)
    #   · disable_atr_stop  =True  (los stops cortaban ganadores: +19pp avg)
    #   · max_holding_days  =120   (Antonacci/Faber dejan correr 4-6 meses)
    #   · target_vol        =0.20  (compromise; 0.25 da +5pp pero +5pp drawdown)
    # Para mode='ml' los defaults se mantienen más conservadores.
    is_trend = (mode == 'trend')
    trend_reverse_exit  = bool(body.get('trend_reverse_exit', is_trend))
    disable_atr_stop    = bool(body.get('disable_atr_stop',   is_trend))
    disable_adx_filter  = bool(body.get('disable_adx_filter', False))
    max_hold_override   = body.get('max_holding_days', 120 if is_trend else None)
    target_vol_override = body.get('target_vol',       0.20 if is_trend else None)

    # ── LLM-gate (opcional, replica el flujo del demo_bot_llm.py) ───────────
    # Cuando está activado, antes de abrir cada nueva posición consulta al
    # LLM (Qwen-0.5B-Instruct local). Si el LLM recomienda la dirección
    # contraria a la del bot, la señal queda vetada. Cache mensual por
    # (símbolo, año-mes) → reduce drásticamente el coste de cómputo.
    use_llm     = bool(body.get('use_llm', False))
    llm_provider = str(body.get('llm_provider', 'local'))
    # Tier 3.3: ml_weight ∈ [0,1] controla la mezcla ML/LLM en el sizing.
    # Default 1.0 = LLM no afecta sizing (sólo veto, comportamiento legacy).
    # Razón: el sweep en compare_hybrid_ratios.py NO encontró ningún ratio
    # con Δ_Sharpe IC95% > 0 vs MED_NOLLM (ver PROGRESS.md Tier 3.3). La
    # mezcla continua refactorizada está disponible pero off por default.
    ml_weight   = float(body.get('ml_weight', 1.0))
    ml_weight   = max(0.0, min(1.0, ml_weight))
    # Tier 4.1: parámetros para sweeps de aflojamiento de filtros + sizing.
    # Todos opt-in: defaults preservan comportamiento histórico.
    disable_golden_cross = bool(body.get('disable_golden_cross', False))
    regime_adx_min_override = body.get('regime_adx_min')   # None ⇒ usa default cfg
    max_concurrent = int(body.get('max_concurrent', 3))
    debug_filter_counts = bool(body.get('debug_filter_counts', False))
    # Tier 4.2 — candidatos C2-C5 (todos opt-in, default off).
    # C2: vol-targeting portfolio-level (Moreira-Muir 2017): escala size por
    # target_vol_annual / realized_vol_portfolio_20d. Acota [0.3, 2.0].
    enable_vol_target_overlay = bool(body.get('enable_vol_target_overlay', False))
    vol_target_overlay_annual = float(body.get('vol_target_overlay_annual', 0.20))
    # C4: volatility filter — skip candidato si vol_anual fuera de [min, max]
    vol_filter_min = body.get('vol_filter_min')   # None ⇒ no aplica
    vol_filter_max = body.get('vol_filter_max')
    # C5: mean-reversion combo — operar MR (RSI<10) cuando ADX < mr_adx_max
    enable_mr_combo = bool(body.get('enable_mr_combo', False))
    mr_adx_max      = float(body.get('mr_adx_max', 18.0))
    # C3: top-N cross-sectional rotation — solo entrar top-N por momentum z-score
    topn_rotation = bool(body.get('topn_rotation', False))
    topn_value    = int(body.get('topn_value', 5))
    # Tier 2.3: bull/bear debate + judge (3× llamadas LLM por decisión).
    # Cache trimestral existente amortiza el coste. Off por default — el
    # gate del harness Tier 1.1 decide su valor neto.
    use_debate  = bool(body.get('use_debate', False))
    # Tier 3.2: trade journal con memoria persistente. Off por default;
    # cuando ON, registra cada trade cerrado en SQLite para retrieval k-NN
    # en sesiones futuras. La integración del retrieval al prompt LLM es
    # phase 2.
    use_journal = bool(body.get('use_journal', False))

    # ── Risk gate determinístico (Tier 1.3) ────────────────────────────────
    # Vol-target portfolio + daily-loss limit + kill-switch DD + caps por
    # símbolo y exposición total. Default off para preservar baseline en
    # comparativas; activar con use_risk_gate=True. Tras validación A/B
    # se promoverá a True por defecto.
    use_risk_gate = bool(body.get('use_risk_gate', False))
    risk_gate_overrides = {
        'target_vol_annual':       body.get('rg_target_vol',       0.15),
        'daily_loss_limit_pct':    body.get('rg_daily_loss',       0.03),
        'kill_switch_dd_pct':      body.get('rg_kill_dd',          0.12),
        'kill_switch_pause_days':  body.get('rg_kill_pause_days',  5),
        'max_position_pct_per_sym':body.get('rg_cap_sym',          0.25),
        'max_total_exposure_pct':  body.get('rg_cap_total',        1.0),
    }

    if not start_date or not end_date:
        return jsonify({'status': 'error', 'error': 'start_date y end_date son obligatorios'}), 400

    try:
        start_dt = pd.Timestamp(start_date)
        end_dt   = pd.Timestamp(end_date)
    except Exception:
        return jsonify({'status': 'error', 'error': 'Formato de fecha inválido (usar YYYY-MM-DD)'}), 400

    if start_dt >= end_dt:
        return jsonify({'status': 'error', 'error': 'start_date debe ser anterior a end_date'}), 400

    today = pd.Timestamp(datetime.now().date())
    if end_dt > today:
        end_dt   = today
        end_date = today.strftime('%Y-%m-%d')

    # ── Filtros de régimen RE-ACTIVADOS + endurecidos ────────────────────────
    # Iteración 1 (Tier 1): trend filter on, crash filter on. Resultado:
    #   long+short -22%→-25%, long-only -22%→-17%. Drawdown -50%→-36%. WR 44%.
    #   Diagnóstico: el ML tiene R²<0 en varios activos (AAPL/TSLA/SOL),
    #   PF<1 → estamos operando ruido. Soluciones de la literatura:
    #
    #   Tier 2 (esta iteración):
    #   · max_holding_days=30 (era 15) → 6× horizon de predicción, deja correr
    #     trends (Turtle System 2: salida 20-55d). Avg hold actual 10.7d → cortamos
    #     TPs prematuramente. PF subirá si avg_win/avg_loss llega al 2:1 diseñado.
    #   · min_signal_strength=0.012 (era 0.005) → suelo del umbral 1.2%. Filtra
    #     señales sub-modelo. Con threshold dinámico ya en 0.043, el suelo es
    #     un cinturón de seguridad para periodos de baja volatilidad.
    #   · regime_adx_min=22 (era 18) → exige tendencia clara, ADX≥22 según el
    #     manual original de Wilder es el mínimo para considerar "trending".
    #   · atr_stop_mult=2.0 mantenido (consistente con TP=2×SL).
    cfg = BotConfig(
        initial_capital=init_cap,
        signal_percentile=sig_pct,
        kelly_scale=kelly_sc,
        allow_short=allow_short,
        commission=commission_rate,
        regime_use_sma200=True,
        regime_crash_filter=True,
        regime_adx_min=(0.0 if disable_adx_filter
                        else (float(regime_adx_min_override) if regime_adx_min_override is not None
                              else 22.0)),
        atr_stop_mult=2.0,
        max_position_pct=0.18,
        target_vol=float(target_vol_override) if target_vol_override is not None else 0.15,
        min_signal_strength=0.012,
        max_holding_days=int(max_hold_override) if max_hold_override is not None else 30,
    )

    logger.info(f"=== Autonomous multi-asset backtest [{mode.upper()}]: {start_date}→{end_date} cap=${init_cap:,.0f} ===")

    try:
        # Tier 4.2 C1: opcional añadir ETFs sectoriales/factor para diversificar
        # el cluster tech (AAPL/MSFT/NVDA correlación >0.7 entre ellos). XLE/XLF
        # baja correlación con tech, GLD anticíclico, MTUM factor momentum,
        # IWM small-cap. Total +5 activos → 15 en universo. Default: off.
        include_etfs = bool(body.get('include_etfs', False))
        watchlist = (
            [('AAPL', 'stock'), ('MSFT', 'stock'), ('NVDA', 'stock'),
             ('TSLA', 'stock'), ('AMZN', 'stock'), ('GOOGL', 'stock')] +
            [('BTC-USD', 'crypto'), ('ETH-USD', 'crypto'),
             ('SOL-USD', 'crypto'), ('BNB-USD', 'crypto')]
        )
        if include_etfs:
            watchlist += [
                ('XLE',  'stock'),  # energy sector — baja corr con tech
                ('XLF',  'stock'),  # financials — baja corr con tech
                ('GLD',  'stock'),  # gold — anticíclico
                ('MTUM', 'stock'),  # momentum factor — diversifica con tilt momentum
                ('IWM',  'stock'),  # small-cap Russell 2000 — diversificación size
            ]

        loader    = FinancialDataLoader()
        processor = DataProcessor()
        # 4 años de entrenamiento (antes 3y) para cubrir múltiples regímenes:
        # bear 2022, recuperación 2023, bull 2024 — y dar al ensemble exposición
        # a cracks/bounces. Más datos → menos overfitting al régimen reciente.
        train_start = (start_dt - pd.DateOffset(years=4)).strftime('%Y-%m-%d')
        dl_end      = (end_dt + pd.DateOffset(days=1)).strftime('%Y-%m-%d')

        # ── 1. Cargar predicciones para cada activo ───────────────────────────
        asset_maps    = {}    # symbol → {date_str → day_data_dict}
        asset_r2      = {}    # symbol → mean_r2 OOS test (info-only, NO usar para gating)
        asset_val_r2  = {}    # symbol → train_validation_r2 (CV pre-test, sí usable para gating)
        trained_ok    = []

        for sym, atype in watchlist:
            try:
                df = loader.download_data(sym, train_start, dl_end, asset_type=atype)
                if df is None or df.empty or len(df) < 100:
                    continue
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df = df[df.index <= end_dt]
                df = processor.clean_data(df)
                df = processor.add_technical_indicators(df)

                if mode == 'trend':
                    # ── Modo trend-follower puro (Faber 2007 / Antonacci dual-mom) ──
                    # No usa ML. Señal = SMA50/SMA200. Se valida SOLO con datos ≤ t,
                    # los SMAs son rolling causales → leak-free por construcción.
                    tdf = df[df.index >= start_dt].copy()
                    date_map = {}
                    for j in range(len(tdf)):
                        idx = tdf.index[j]
                        ds  = str(idx.date()) if hasattr(idx, 'date') else str(idx)[:10]
                        sma50  = tdf['SMA_50'].iloc[j]   if 'SMA_50'  in tdf.columns else np.nan
                        sma200 = tdf['SMA_200'].iloc[j]  if 'SMA_200' in tdf.columns else np.nan
                        if pd.isna(sma50) or pd.isna(sma200) or sma200 <= 0:
                            continue
                        # trend_strength positivo = uptrend, negativo = downtrend
                        trend_strength = float(sma50 / sma200 - 1.0)
                        # Mantener mínimo 0.015 de magnitud para pasar los gates
                        # de la fase de selección (>= 0.008). Conf 0.55-0.85 según fuerza.
                        sign_t = 1.0 if trend_strength >= 0 else -1.0
                        pred_t = sign_t * max(abs(trend_strength), 0.015)
                        conf_t = 0.55 + min(abs(trend_strength) * 5.0, 0.30)
                        date_map[ds] = {
                            'pred':       pred_t,
                            'conf':       conf_t,
                            'close':      float(tdf['Close'].iloc[j]),
                            'low':        float(tdf['Low'].iloc[j])  if 'Low'  in tdf.columns else float(tdf['Close'].iloc[j]),
                            'high':       float(tdf['High'].iloc[j]) if 'High' in tdf.columns else float(tdf['Close'].iloc[j]),
                            'atr':        float(tdf['ATR'].iloc[j])        if ('ATR'        in tdf.columns and not pd.isna(tdf['ATR'].iloc[j]))        else float(tdf['Close'].iloc[j]) * 0.015,
                            'sma200':     float(sma200),
                            'sma50':      float(sma50),
                            'adx':        float(tdf['ADX'].iloc[j])        if ('ADX'        in tdf.columns and not pd.isna(tdf['ADX'].iloc[j]))        else None,
                            'volatility': float(tdf['Volatility'].iloc[j]) if ('Volatility' in tdf.columns and not pd.isna(tdf['Volatility'].iloc[j])) else 0.20,
                        }
                    if date_map:
                        asset_maps[sym]   = date_map
                        asset_r2[sym]     = 1.0   # neutro: sin ML que evaluar
                        asset_val_r2[sym] = 1.0   # always-pass del R² gate en trend mode
                        trained_ok.append(sym)
                        logger.info(f"  {sym}: {len(date_map)} días test · TREND-MODE (SMA50/SMA200)")
                    continue  # siguiente activo

                # ── Modo ML (default) ────────────────────────────────────────────
                preds = _generate_bot_predictions_by_date(df, atype, start_date)
                n_p   = len(preds['predictions'])
                t_idx = preds['test_start_idx']
                tdf   = df.iloc[t_idx: t_idx + n_p].copy()

                date_map = {}
                for j in range(min(n_p, len(tdf))):
                    idx = tdf.index[j]
                    ds  = str(idx.date()) if hasattr(idx, 'date') else str(idx)[:10]
                    date_map[ds] = {
                        'pred':       float(preds['predictions'][j]),
                        'conf':       float(preds['confidences'][j]),
                        'close':      float(tdf['Close'].iloc[j]),
                        'low':        float(tdf['Low'].iloc[j])  if 'Low'  in tdf.columns else float(tdf['Close'].iloc[j]),
                        'high':       float(tdf['High'].iloc[j]) if 'High' in tdf.columns else float(tdf['Close'].iloc[j]),
                        'atr':        float(tdf['ATR'].iloc[j])        if ('ATR'        in tdf.columns and not pd.isna(tdf['ATR'].iloc[j]))        else float(tdf['Close'].iloc[j]) * 0.015,
                        'sma200':     float(tdf['SMA_200'].iloc[j])    if ('SMA_200'    in tdf.columns and not pd.isna(tdf['SMA_200'].iloc[j]))    else None,
                        'sma50':      float(tdf['SMA_50'].iloc[j])     if ('SMA_50'     in tdf.columns and not pd.isna(tdf['SMA_50'].iloc[j]))     else None,
                        'adx':        float(tdf['ADX'].iloc[j])        if ('ADX'        in tdf.columns and not pd.isna(tdf['ADX'].iloc[j]))        else None,
                        'volatility': float(tdf['Volatility'].iloc[j]) if ('Volatility' in tdf.columns and not pd.isna(tdf['Volatility'].iloc[j])) else 0.20,
                    }

                if date_map:
                    asset_maps[sym]   = date_map
                    asset_r2[sym]     = float(preds.get('mean_r2', 0.0))
                    asset_val_r2[sym] = float(preds.get('train_validation_r2', 0.0))
                    trained_ok.append(sym)
                    logger.info(
                        f"  {sym}: {len(date_map)} días test · "
                        f"val_r2={asset_val_r2[sym]:+.4f} · oos_r2={asset_r2[sym]:+.4f}"
                    )
            except Exception as ex:
                logger.warning(f"  Skip {sym}: {ex}")

        if not asset_maps:
            return jsonify({'status': 'error', 'error': 'No se pudieron cargar datos para ningún activo'}), 400

        # ── Filtro de régimen de mercado: SPY vs SMA_200 (Faber 2007) ──────────
        # Antes usábamos SMA_50 → demasiados flips ruidosos al rozar la media.
        # SMA_200 es el filtro de tendencia secular: SPY > SMA200 → mercado alcista
        # estructural, sólo deben tomarse posiciones LONG en stocks (los SHORTs
        # contra una tendencia secular son guaranteed bleed). En bear (SPY<SMA200)
        # permitimos ambos sentidos pero sólo LONG si el activo tiene su propio
        # SMA200 alcista, y SHORT si lo tiene bajista (combinación con per-asset).
        # Crypto se rige por su propio SMA_200 individual (BTC y SPY descorrelacionados).
        spy_regime = {}   # ds → True = SPY > SMA200 (bull), False = bear
        try:
            spy_df_raw = loader.download_data('SPY', train_start, dl_end, asset_type='stock')
            if spy_df_raw is not None and not spy_df_raw.empty:
                if hasattr(spy_df_raw.index, 'tz') and spy_df_raw.index.tz is not None:
                    spy_df_raw.index = spy_df_raw.index.tz_localize(None)
                spy_close   = spy_df_raw['Close']
                spy_sma200  = spy_close.rolling(200, min_periods=100).mean()
                for i in range(len(spy_df_raw)):
                    ds_i = str(spy_df_raw.index[i].date())
                    if not pd.isna(spy_sma200.iloc[i]):
                        spy_regime[ds_i] = bool(spy_close.iloc[i] >= spy_sma200.iloc[i])
                bull_days = sum(1 for v in spy_regime.values() if v)
                logger.info(f"SPY regime: {len(spy_regime)} días ({bull_days} bull, {len(spy_regime)-bull_days} bear)")
        except Exception as spy_err:
            logger.warning(f"SPY regime filter omitido: {spy_err}")

        # ── 2. Secuencia de fechas unificada ──────────────────────────────────
        all_dates = sorted({ds for dm in asset_maps.values() for ds in dm})
        if len(all_dates) < 5:
            return jsonify({'status': 'error', 'error': 'Período de test demasiado corto'}), 400

        # Umbral global de señal — Red Team Finding 5: calibrar SOLO en los
        # primeros 30 días del test para evitar look-ahead bias (usar datos
        # futuros para calcular el umbral del día 1 crea sesgo de información).
        calib_dates = set(all_dates[:min(30, len(all_dates))])
        calib_ss = [abs(d['pred']) * d['conf']
                    for dm in asset_maps.values()
                    for ds, d in dm.items() if ds in calib_dates]
        if len(calib_ss) < 5:
            # fallback: usar todos los datos si el período es muy corto
            calib_ss = [abs(d['pred']) * d['conf']
                        for dm in asset_maps.values() for d in dm.values()]
        raw_thr = float(np.percentile(calib_ss, cfg.signal_percentile * 100))
        threshold = max(raw_thr, cfg.min_signal_strength)

        # ── LLM client + cache (trimestral por símbolo) ──────────────────────
        # Cache trimestral (YYYY-Qn) en vez de mensual: ~3× menos llamadas en
        # backtests largos sin perder excesivo poder de gating.
        llm = None
        llm_cache = {}     # (sym, 'YYYY-Qn') → recommendation_str
        llm_stats = {'calls': 0, 'cache_hits': 0, 'errors': 0, 'vetoed': 0,
                     'budget_skipped': 0, 'total_time': 0.0,
                     'sent_real_news': 0, 'sent_implied_only': 0}
        # Tier 4.1: contadores opt-in de filtros (solo si debug_filter_counts=True).
        filter_counts = {
            'total_evaluated': 0, 'signal_percentile': 0, 'quality_gate': 0,
            'asset_r2_gate': 0, 'no_direction': 0, 'sma200_filter': 0,
            'golden_cross': 0, 'adx_filter': 0, 'crash_filter': 0,
            'spy_regime': 0,
        }
        # Budget: si se supera, no se llama más al LLM y se devuelve HOLD por defecto.
        # Esto evita que un backtest se quede colgado durante horas si hay muchos candidatos.
        llm_budget = int(body.get('llm_budget', 60))
        if use_llm:
            try:
                from utils.llm_client import LLMClient
                llm = LLMClient(provider=llm_provider)
                # Warm-up: una llamada barata para cargar el modelo
                t_warm = time.time()
                _ = llm.analyze_sentiment("Test market warmup signal.")
                logger.info(f"LLM warm-up OK ({time.time()-t_warm:.1f}s) · budget={llm_budget}")
            except Exception as llm_err:
                logger.warning(f"LLM gate desactivado por error: {llm_err}")
                llm = None
                use_llm = False

        # ── News cache (Tier 1.2) ─────────────────────────────────────────
        # Reemplaza el hardcode 'sent = "Neutral..."' por implied sentiment
        # desde mercado + (si hay) noticias reales del cache SQLite.
        news_cache = None
        try:
            from utils.news_cache import NewsCache, build_sentiment_string
            news_cache = NewsCache()
        except Exception as nc_err:
            logger.warning(f"News cache desactivado: {nc_err}")
            build_sentiment_string = None  # type: ignore

        # ── Risk gate determinístico (Tier 1.3) ───────────────────────────
        risk_gate = None
        if use_risk_gate:
            try:
                from trading_bot.risk_gate import RiskGate, RiskGateConfig
                risk_gate = RiskGate(RiskGateConfig(**risk_gate_overrides))
                logger.info(f"Risk gate ON: target_vol={risk_gate.cfg.target_vol_annual} "
                            f"daily_loss={risk_gate.cfg.daily_loss_limit_pct} "
                            f"kill_dd={risk_gate.cfg.kill_switch_dd_pct} "
                            f"cap_sym={risk_gate.cfg.max_position_pct_per_sym}")
            except Exception as rg_err:
                logger.warning(f"Risk gate desactivado por error: {rg_err}")
                risk_gate = None

        # ── Trade journal (Tier 3.2) ──────────────────────────────────────
        trade_journal = None
        if use_journal:
            try:
                from utils.trade_journal import TradeJournal
                trade_journal = TradeJournal()
                logger.info(f"Trade journal ON: db={trade_journal.db_path}  "
                            f"existing_trades={trade_journal.count()}")
            except Exception as tj_err:
                logger.warning(f"Trade journal desactivado por error: {tj_err}")
                trade_journal = None

        def _quarter_key(ds: str) -> str:
            # ds = 'YYYY-MM-DD' → 'YYYY-Q{1,2,3,4}'
            try:
                y, m, _ = ds.split('-')
                q = (int(m) - 1) // 3 + 1
                return f"{y}-Q{q}"
            except Exception:
                return ds[:7]

        def _llm_decision(sym: str, ds: str, day: dict) -> dict:
            """Devuelve {'rec': str, 'confidence': float in [0,1]} con cache trimestral.

            Tier 3.3 refactor: devuelve tupla (rec, confidence_continuous) en
            lugar de sólo string. La confianza alimenta el _compute_hybrid_score
            aguas abajo para modular sizing — vs el veto binario anterior.
            """
            if not use_llm or llm is None:
                return {'rec': '', 'confidence': 0.5}
            cache_k = (sym, _quarter_key(ds))
            if cache_k in llm_cache:
                llm_stats['cache_hits'] += 1
                return llm_cache[cache_k]
            # Budget: dejar pasar (HOLD) sin llamar al modelo
            if llm_stats['calls'] >= llm_budget:
                llm_stats['budget_skipped'] += 1
                out = {'rec': 'HOLD', 'confidence': 0.5}
                llm_cache[cache_k] = out
                return out
            close = day.get('close', 0.0)
            sma50  = day.get('sma50')
            sma200 = day.get('sma200')
            atr    = day.get('atr', 0.0)
            adx    = day.get('adx')
            vol    = day.get('volatility', 0.0) * 100
            pred   = day.get('pred', 0.0)
            conf   = day.get('conf', 0.5)
            if sma50 and sma200 and sma200 > 0:
                ratio = sma50 / sma200 - 1
                trend_label = 'uptrend' if ratio > 0 else 'downtrend'
                trend_str = f"SMA50/SMA200 = {sma50/sma200:.3f} ({trend_label}, gap {ratio*100:+.1f}%)"
            else:
                trend_str = "tendencia indeterminada"
            tech = (f"Precio ${close:,.2f}. {trend_str}. ATR ${atr:,.2f}. "
                    f"ADX {f'{adx:.1f}' if adx is not None else 'n/a'}. "
                    f"Volatilidad anualizada {vol:.1f}%.")
            # ── Sentiment (Tier 1.2): implied desde retornos + cache real opcional ──
            sent = "Neutral (sin feed de noticias en backtest)"
            if build_sentiment_string is not None:
                try:
                    date_map = asset_maps.get(sym, {})
                    prior_dates = sorted(d for d in date_map if d < ds)
                    def _ret_pct(n: int) -> float:
                        if len(prior_dates) < n or close <= 0:
                            return 0.0
                        ref = date_map.get(prior_dates[-n], {}).get('close', close)
                        return (close / ref - 1.0) * 100 if ref > 0 else 0.0
                    ret_5d  = _ret_pct(5)
                    ret_20d = _ret_pct(20)
                    atr_pct = (atr / close * 100) if close > 0 else None
                    sent, sent_label, n_real = build_sentiment_string(
                        symbol=sym,
                        current_date=ds,
                        ret_5d_pct=ret_5d,
                        ret_20d_pct=ret_20d,
                        rsi=day.get('rsi'),
                        sma50=sma50,
                        sma200=sma200,
                        vol_pct=vol,
                        atr_pct_of_close=atr_pct,
                        news_cache=news_cache,
                    )
                    if n_real > 0:
                        llm_stats['sent_real_news'] += 1
                    else:
                        llm_stats['sent_implied_only'] += 1
                except Exception as se:
                    logger.debug(f"sentiment build error {sym} {ds}: {se}")
            risk = (f"Volatilidad {vol:.1f}%. Riesgo "
                    f"{'ALTO' if vol > 40 else 'MEDIO' if vol > 20 else 'BAJO'}.")
            ml_sum = (f"Señal trend-follower: pred={pred:+.4f} "
                      f"({'LONG' if pred > 0 else 'SHORT'}), confianza={conf:.2f}.")
            t0 = time.time()
            llm_conf = 0.5
            try:
                # ── Tier 2.3: bull/bear debate + judge (opt-in) ─────────
                if use_debate:
                    from agents.debate import run_debate, is_veto_from_debate
                    deb = run_debate(llm, ml_sum, tech, sent, risk)
                    llm_stats['calls'] += 3 if deb.get('ok') else 2
                    llm_stats['total_time'] += time.time() - t0
                    cm = float(deb.get('confidence_mult', 0.5))
                    direc_label = 'LONG' if (day.get('pred', 0.0) > 0) else 'SHORT'
                    if is_veto_from_debate(direc_label, cm, threshold=0.4):
                        rec = 'HOLD'
                    else:
                        rec = 'BUY' if direc_label == 'LONG' else 'SELL'
                    # Tier 3.3: confidence_mult del judge ∈ [0.3, 1.0] mapea
                    # razonablemente a [0, 1] para el hybrid score.
                    llm_conf = float(np.clip(cm, 0.0, 1.0))
                    llm_stats.setdefault('debate_runs', 0)
                    llm_stats['debate_runs'] += 1
                    llm_stats.setdefault('debate_avg_conf_mult_sum', 0.0)
                    llm_stats['debate_avg_conf_mult_sum'] += cm
                else:
                    mkt = llm.interpret_market_data(tech, sent, risk, ml_sum)
                    parsed = mkt.get('parsed', {}) if isinstance(mkt, dict) else {}
                    rec = str(parsed.get('recommendation', 'HOLD')).upper().strip()
                    if '|' in rec:
                        rec = 'HOLD'
                    # Tier 3.3: extraer confianza continua del JSON parseado
                    try:
                        llm_conf = float(parsed.get('confidence', 0.5))
                        llm_conf = float(np.clip(llm_conf, 0.0, 1.0))
                    except Exception:
                        llm_conf = 0.5
                    llm_stats['calls'] += 1
                    llm_stats['total_time'] += time.time() - t0
            except Exception as e:
                logger.debug(f"LLM call error {sym} {ds}: {e}")
                llm_stats['errors'] += 1
                rec = 'HOLD'
                llm_conf = 0.5
            out = {'rec': rec, 'confidence': llm_conf}
            llm_cache[cache_k] = out
            return out

        # ── 3. Simulación multi-activo ────────────────────────────────────────
        def _kelly(pred, conf, stop_pct):
            p = float(np.clip(conf, 0.15, 0.85))
            risk = max(float(stop_pct), 0.005)
            # TP = 2×|pred| → b = 2×|pred|/risk; mínimo 0.5 para que conf≥0.67 → Kelly>0
            b = max(2 * abs(float(pred)) / risk, 0.5)
            return float(np.clip((p * b - (1 - p)) / b, 0.0, 1.0))

        MAX_CONCURRENT = max(1, int(max_concurrent))   # Tier 4.1: parametrizable (default 3)

        capital     = cfg.initial_capital
        positions   = {}   # sym → position_dict (múltiples posiciones simultáneas)
        equity_vals = [capital]
        equity_dts  = [all_dates[0]]
        all_trades  = []
        all_ss      = []
        active_syms_history = {}  # ds → [sym, ...]

        _t_loop_start = time.time()
        _log_every = max(len(all_dates) // 8, 1)  # ≈8 avisos por backtest

        for _di, ds in enumerate(all_dates):
            if use_llm and _di > 0 and _di % _log_every == 0:
                logger.info(
                    f"  · día {_di}/{len(all_dates)} ({ds}) "
                    f"LLM calls={llm_stats['calls']} hits={llm_stats['cache_hits']} "
                    f"vetoed={llm_stats['vetoed']} skip={llm_stats['budget_skipped']} "
                    f"({time.time()-_t_loop_start:.0f}s)"
                )

            # ── Tier 1.3: hook on_day_start del risk gate ─────────────────────
            # Computa equity actual para que el gate sepa peak/DD/daily-pnl.
            # Si dispara kill-switch → cerramos todas las posiciones al close.
            risk_force_liquidate = False
            risk_paused = False
            risk_daily_lock = False
            if risk_gate is not None:
                # equity provisional: cash + valor de posiciones a precio de ayer
                # (mark-to-market real se hace en paso 3 del loop). Para el gate
                # usamos el último equity_vals registrado.
                eq_now = equity_vals[-1] if equity_vals else cfg.initial_capital
                rg_status = risk_gate.on_day_start(ds, eq_now)
                risk_force_liquidate = bool(rg_status.get('force_liquidate'))
                risk_paused = bool(rg_status.get('paused'))
                risk_daily_lock = bool(rg_status.get('daily_lock'))
                if risk_force_liquidate:
                    logger.info(f"[risk_gate] KILL-SWITCH @ {ds}  DD={rg_status['current_dd']:.2%}  → liquidate all + pause")

            # ── 1. Gestión de posiciones abiertas ─────────────────────────────
            for sym in list(positions.keys()):
                position = positions[sym]
                if sym not in asset_maps or ds not in asset_maps[sym]:
                    continue
                day  = asset_maps[sym][ds]
                atr  = day['atr']
                hold = (pd.Timestamp(ds) - pd.Timestamp(position['entry_date'])).days

                # SL fijo: no se mueve durante la vida de la posición.
                # Breakeven eliminado: generaba muchas micro-ganancias (0.25 ATR)
                # que hundían el avg_win a ~= avg_loss → PF < 1 aunque WR > 47%.
                # Con SL=1.5 ATR y TP=3 ATR (R:R 2:1) el sistema es rentable
                # teóricamente con WR > 33.3%. Dejar que los ganadores corran.

                exit_p = None; exit_r = None
                # ── Tier 1.3: kill-switch del risk_gate cierra TODO al close ──
                if risk_force_liquidate:
                    exit_p = day['close']; exit_r = 'KILL_SWITCH'
                # ── Reverse exit (trend-follow): si la señal cambia de signo
                #    respecto al tipo de la posición, salir al close. En trend
                #    mode esto = death cross (LONG) o golden cross (SHORT).
                if exit_p is None and trend_reverse_exit:
                    pred_t = day.get('pred', 0.0)
                    if (position['type'] == 'LONG' and pred_t < 0) or \
                       (position['type'] == 'SHORT' and pred_t > 0):
                        exit_p = day['close']; exit_r = 'TREND_REVERSE'
                # ── ATR stops (sólo si están habilitados) ────────────────────
                if exit_p is None and not disable_atr_stop:
                    if position['type'] == 'LONG':
                        sl_hit = day['low']  <= position['stop_loss']
                        tp_hit = day['high'] >= position['take_profit']
                        if sl_hit and tp_hit:
                            # Ambos niveles tocados el mismo día: usar close vs midpoint
                            # para estimar cuál se tocó primero (sin datos intradiarios).
                            mid = (position['stop_loss'] + position['take_profit']) / 2
                            if day['close'] >= mid:
                                exit_p = position['take_profit']; exit_r = 'TAKE_PROFIT'
                            else:
                                exit_p = position['stop_loss'];   exit_r = 'STOP_LOSS'
                        elif sl_hit:
                            exit_p = position['stop_loss'];   exit_r = 'STOP_LOSS'
                        elif tp_hit:
                            exit_p = position['take_profit']; exit_r = 'TAKE_PROFIT'
                    else:
                        sl_hit = day['high'] >= position['stop_loss']
                        tp_hit = day['low']  <= position['take_profit']
                        if sl_hit and tp_hit:
                            mid = (position['stop_loss'] + position['take_profit']) / 2
                            if day['close'] <= mid:
                                exit_p = position['take_profit']; exit_r = 'TAKE_PROFIT'
                            else:
                                exit_p = position['stop_loss'];   exit_r = 'STOP_LOSS'
                        elif sl_hit:
                            exit_p = position['stop_loss'];   exit_r = 'STOP_LOSS'
                        elif tp_hit:
                            exit_p = position['take_profit']; exit_r = 'TAKE_PROFIT'

                if exit_p is None and hold >= cfg.max_holding_days:
                    exit_p = day['close']; exit_r = 'MAX_HOLD'
                # SIGNAL exit eliminado: dejamos que TP/SL/MAX_HOLD gestionen la salida.
                # El exit anticipado por señal contraria cortaba los ganadores antes de
                # alcanzar el TP y resultaba en un profit factor < 1.

                if exit_p is not None:
                    adj      = -cfg.slippage if position['type'] == 'LONG' else cfg.slippage
                    exit_adj = exit_p * (1 + adj)
                    comm     = exit_adj * position['shares'] * cfg.commission
                    if position['type'] == 'LONG':
                        proceeds = exit_adj * position['shares'] - comm
                    else:
                        proceeds = position['cost_basis'] * 2 - exit_adj * position['shares'] - comm
                    pnl     = proceeds - position['cost_basis']
                    capital += proceeds
                    ret_pct = float(pnl / (position['cost_basis'] + 1e-8) * 100)
                    all_trades.append({
                        'symbol': sym, 'type': position['type'],
                        'action': position['type'],
                        'entry_date': position['entry_date'], 'exit_date': ds,
                        'entry_price': round(position['entry_price'], 4),
                        'exit_price':  round(float(exit_adj), 4),
                        'pnl':         round(float(pnl), 2),
                        'return_pct':  round(ret_pct, 2),
                        'actual_return_pct': round(ret_pct, 2),
                        'predicted_return_pct': 0.0,
                        'exit_reason': exit_r,
                        'hold_days':   max(hold, 0),
                    })
                    # Tier 3.2: persistir el trade cerrado en el journal
                    if trade_journal is not None:
                        try:
                            entry_feats = position.get('entry_features') or []
                            trade_journal.log_trade(
                                symbol=sym,
                                entry_date=position['entry_date'],
                                exit_date=ds,
                                action=position['type'],
                                features_vec=entry_feats,
                                pnl_pct=ret_pct,
                                hold_days=max(hold, 0),
                                regime=position.get('regime', ''),
                                rationale=str(exit_r or '')[:500],
                            )
                        except Exception as tj_e:
                            logger.debug(f"trade_journal.log_trade failed: {tj_e}")
                    del positions[sym]

            # ── 2. Buscar nuevas señales si hay slots libres ───────────────────
            # Tier 1.3: si el risk gate está en pausa post-kill o en lock por
            # daily-loss, NO abrimos nuevas entradas (sólo gestionamos cierres).
            n_slots = MAX_CONCURRENT - len(positions)
            if risk_gate is not None and (risk_paused or risk_daily_lock or risk_force_liquidate):
                n_slots = 0
            if n_slots > 0 and capital > 100:
                candidates = []
                for sym, dm in asset_maps.items():
                    if ds not in dm or sym in positions:
                        continue
                    d  = dm[ds]
                    ss = abs(d['pred']) * d['conf']
                    all_ss.append(ss)
                    if debug_filter_counts: filter_counts['total_evaluated'] += 1
                    if ss <= threshold:
                        if debug_filter_counts: filter_counts['signal_percentile'] += 1
                        continue
                    # ── Quality gate per-señal ────────────────────────────────────
                    if abs(d['pred']) < 0.008 or d['conf'] < 0.45:
                        if debug_filter_counts: filter_counts['quality_gate'] += 1
                        continue
                    # ── Gate de calidad por activo (R² CV pre-test, leak-free) ───
                    if asset_val_r2.get(sym, 0.0) < -0.30:
                        if debug_filter_counts: filter_counts['asset_r2_gate'] += 1
                        continue
                    direc = ('LONG' if d['pred'] > 0 else ('SHORT' if d['pred'] < 0 and cfg.allow_short else None))
                    if direc is None:
                        if debug_filter_counts: filter_counts['no_direction'] += 1
                        continue
                    # Filtro SMA_200 (precio vs su propia media de 200d)
                    if cfg.regime_use_sma200 and d['sma200'] is not None:
                        if direc == 'LONG'  and d['close'] < d['sma200']:
                            if debug_filter_counts: filter_counts['sma200_filter'] += 1
                            continue
                        if direc == 'SHORT' and d['close'] > d['sma200']:
                            if debug_filter_counts: filter_counts['sma200_filter'] += 1
                            continue
                    # ── Confirmación de Golden Cross (SMA50 vs SMA200) ────────────
                    # Tier 4.1: opt-out vía disable_golden_cross para sweeps.
                    if (not disable_golden_cross
                            and d['sma50'] is not None and d['sma200'] is not None):
                        golden = d['sma50'] > d['sma200']
                        if direc == 'LONG'  and not golden:
                            if debug_filter_counts: filter_counts['golden_cross'] += 1
                            continue
                        if direc == 'SHORT' and golden:
                            if debug_filter_counts: filter_counts['golden_cross'] += 1
                            continue
                    # Filtro ADX (tendencia mínima requerida)
                    # Tier 4.2 C5: MR combo — si enable_mr_combo y ADX bajo (lateral)
                    # PERO RSI<30 (oversold), permitir entrada como mean-reversion.
                    # RobotWealth + Price Action Lab: RSI<30 en mercado lateral con
                    # SMA200 alcista tiene edge documentado.
                    if d['adx'] is not None and d['adx'] < cfg.regime_adx_min:
                        rsi_val = d.get('rsi')
                        mr_pass = (enable_mr_combo
                                   and direc == 'LONG'
                                   and d['adx'] < mr_adx_max
                                   and rsi_val is not None and rsi_val < 30)
                        if not mr_pass:
                            if debug_filter_counts: filter_counts['adx_filter'] += 1
                            continue
                    # Filtro de crash: evitar LONG tras caída >5% en 5 días
                    if cfg.regime_crash_filter and direc == 'LONG':
                        sorted_sym_dates = sorted(k for k in dm if k <= ds)
                        if len(sorted_sym_dates) >= 6:
                            p5d_ago = dm[sorted_sym_dates[-6]]['close']
                            if p5d_ago > 0 and (d['close'] / p5d_ago - 1) < cfg.regime_crash_threshold:
                                if debug_filter_counts: filter_counts['crash_filter'] += 1
                                continue
                    # Filtro de régimen SPY (sólo aplica a STOCKS).
                    is_stock = not sym.endswith('-USD')
                    if spy_regime and is_stock:
                        market_up = spy_regime.get(ds)
                        if market_up is not None:
                            if direc == 'SHORT' and market_up:
                                if debug_filter_counts: filter_counts['spy_regime'] += 1
                                continue
                            if direc == 'LONG'  and not market_up:
                                if debug_filter_counts: filter_counts['spy_regime'] += 1
                                continue
                    # ── LLM gate (último gate, sólo si está activado) ────────────
                    # Tier 3.3: el LLM ya no es sólo veto binario — devuelve
                    # (rec, confidence_continuous). El veto se mantiene
                    # cuando contradice fuerte; cuando coincide o es neutral,
                    # la confianza alimenta el sizing aguas abajo.
                    llm_dec_for_sizing = None
                    if use_llm:
                        ld = _llm_decision(sym, ds, d)
                        rec = ld.get('rec', '') if isinstance(ld, dict) else str(ld)
                        if direc == 'LONG' and rec in ('SELL', 'STRONG_SELL', 'VENDER', 'VENTA', 'VENTA_FUERTE'):
                            llm_stats['vetoed'] += 1
                            continue
                        if direc == 'SHORT' and rec in ('BUY', 'STRONG_BUY', 'COMPRAR', 'COMPRA', 'COMPRA_FUERTE'):
                            llm_stats['vetoed'] += 1
                            continue
                        llm_dec_for_sizing = ld if isinstance(ld, dict) else None
                    # ── Tier 4.2 C3: cross-sectional momentum 60d para ranking ──
                    mom60 = 0.0
                    if topn_rotation:
                        sorted_d = sorted(k for k in dm if k <= ds)
                        if len(sorted_d) >= 61:
                            p_now = d['close']
                            p_60  = dm[sorted_d[-61]]['close']
                            if p_60 > 0:
                                mom60 = (p_now / p_60 - 1.0)
                    candidates.append((ss, sym, d, direc, llm_dec_for_sizing, mom60))

                # ── Tier 4.2 C3: ranking
                #   default → por signal strength (ss)
                #   topn_rotation → por momentum 60d desc, tomar top-N
                if topn_rotation:
                    candidates.sort(key=lambda x: x[5], reverse=True)
                    effective_slots = min(n_slots, topn_value)
                    candidates = candidates[:topn_value]
                else:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    effective_slots = n_slots
                for ss, sym, bday, direc, llm_dec, _mom60 in candidates[:effective_slots]:
                    stop_pct = (cfg.atr_stop_mult * bday['atr']) / max(bday['close'], 1e-8)
                    # ── Tier 3.3: hybrid confidence (ml_weight × ml_conf + (1-w) × llm_conf) ──
                    # Sólo si LLM activo y devolvió confidence_continuous. La
                    # dirección sigue siendo 100% del ML (no flippeada por LLM).
                    base_ml_conf = float(bday.get('conf', 0.5))
                    if use_llm and llm_dec is not None:
                        llm_conf = float(llm_dec.get('confidence', 0.5))
                        # rec neutral (HOLD/WAIT) penaliza ligeramente; rec
                        # alineado con ML deja confianza intacta o la sube.
                        rec = str(llm_dec.get('rec', '')).upper()
                        if rec in ('HOLD', 'WAIT', 'MANTENER', ''):
                            llm_conf *= 0.7   # neutral = menos convicción
                        eff_conf = ml_weight * base_ml_conf + (1.0 - ml_weight) * llm_conf
                    else:
                        eff_conf = base_ml_conf
                    kelly    = _kelly(bday['pred'], eff_conf, stop_pct)
                    rel_str  = min(ss / max(threshold, 1e-9), 3.0)
                    vol_sc   = float(np.clip(cfg.target_vol / max(bday['volatility'], 0.05), 0.3, 2.0))
                    size_pct = min(kelly * rel_str * cfg.kelly_scale * vol_sc, cfg.max_position_pct)
                    # ── Tier 4.2 C2: vol-target overlay portfolio-level (Moreira-Muir 2017) ──
                    if enable_vol_target_overlay and len(equity_vals) >= 21:
                        # realized portfolio vol annualizada de los últimos 20d
                        eq_arr = np.array(equity_vals[-21:], dtype=float)
                        log_rets = np.diff(np.log(eq_arr + 1e-10))
                        realized_vol = float(np.std(log_rets) * np.sqrt(252))
                        if realized_vol > 0.001:
                            overlay_mult = float(np.clip(
                                vol_target_overlay_annual / realized_vol, 0.3, 2.0))
                            size_pct = min(size_pct * overlay_mult, cfg.max_position_pct)
                    # ── Tier 4.2 C4: volatility filter — skip si fuera del rango ──
                    if vol_filter_min is not None and bday['volatility'] < float(vol_filter_min):
                        continue
                    if vol_filter_max is not None and bday['volatility'] > float(vol_filter_max):
                        continue
                    # ── Tier 1.3: aplicar caps + vol-target overlay del risk_gate ──
                    if risk_gate is not None:
                        eq_now = equity_vals[-1] if equity_vals else cfg.initial_capital
                        open_val_total = sum(
                            (positions[s]['shares'] * asset_maps[s][ds]['close'])
                            if positions[s]['type'] == 'LONG'
                            else (positions[s]['cost_basis'] - positions[s]['shares'] * asset_maps[s][ds]['close'])
                            for s in positions
                            if s in asset_maps and ds in asset_maps[s]
                        )
                        allowed, size_pct, _reason = risk_gate.gate_new_position(
                            ds=ds, equity=eq_now,
                            open_position_value_total=open_val_total,
                            sym=sym, candidate_size_pct=size_pct,
                            equity_vals=equity_vals,
                        )
                        if not allowed:
                            continue
                    pos_val  = capital * size_pct
                    if pos_val < 100:
                        continue
                    entry_adj = bday['close'] * (1 + (cfg.slippage if direc == 'LONG' else -cfg.slippage))
                    shares    = pos_val / entry_adj
                    capital  -= pos_val + entry_adj * shares * cfg.commission
                    sl_dist   = cfg.atr_stop_mult * bday['atr']
                    sl = entry_adj - sl_dist if direc == 'LONG' else entry_adj + sl_dist
                    tp = entry_adj + 2.0 * sl_dist if direc == 'LONG' else entry_adj - 2.0 * sl_dist  # R:R 2:1 (TP=3 ATR vs SL=1.5 ATR)
                    positions[sym] = {
                        'symbol': sym, 'type': direc, 'entry_date': ds,
                        'entry_price': entry_adj, 'shares': shares,
                        'cost_basis': pos_val, 'stop_loss': sl, 'take_profit': tp,
                        # Tier 3.2: features compactos al entry para retrieval k-NN futuro
                        'entry_features': [
                            float(bday.get('pred', 0.0)),
                            float(bday.get('conf', 0.5)),
                            float((bday.get('sma50', 0.0) or 0.0) / max(bday.get('sma200', 1.0) or 1.0, 1e-6) - 1.0),
                            float((bday.get('atr', 0.0) or 0.0) / max(bday.get('close', 1.0) or 1.0, 1e-6)),
                            float((bday.get('adx', 0.0) or 0.0) / 100.0),
                            float(bday.get('volatility', 0.0) or 0.0),
                            float(rel_str),
                            float(size_pct),
                        ],
                        'regime': ('bull' if (bday.get('sma50', 0) > bday.get('sma200', 0))
                                   else 'bear'),
                    }

            # ── 3. Mark-to-market ──────────────────────────────────────────────
            port_val = capital
            for sym, pos in positions.items():
                if sym in asset_maps and ds in asset_maps[sym]:
                    cp = asset_maps[sym][ds]['close']
                    if pos['type'] == 'LONG':
                        port_val += pos['shares'] * cp
                    else:
                        port_val += pos['cost_basis'] - pos['shares'] * cp
            equity_vals.append(port_val)
            equity_dts.append(ds)
            active_syms_history[ds] = list(positions.keys())

        # ── Cerrar todas las posiciones abiertas al final del test ─────────────
        last_ds = all_dates[-1]
        for sym, position in list(positions.items()):
            if sym in asset_maps and last_ds in asset_maps[sym]:
                lc   = asset_maps[sym][last_ds]['close']
                adj  = -cfg.slippage if position['type'] == 'LONG' else cfg.slippage
                ep   = lc * (1 + adj)
                comm = ep * position['shares'] * cfg.commission
                if position['type'] == 'LONG':
                    proceeds = ep * position['shares'] - comm
                else:
                    proceeds = position['cost_basis'] * 2 - ep * position['shares'] - comm
                pnl        = proceeds - position['cost_basis']
                capital   += proceeds
                final_hold = (pd.Timestamp(last_ds) - pd.Timestamp(position['entry_date'])).days
                final_ret  = float(pnl / (position['cost_basis'] + 1e-8) * 100)
                all_trades.append({
                    'symbol': sym, 'type': position['type'],
                    'action': position['type'],
                    'entry_date': position['entry_date'], 'exit_date': last_ds,
                    'entry_price': round(position['entry_price'], 4),
                    'exit_price':  round(float(ep), 4),
                    'pnl':         round(float(pnl), 2),
                    'return_pct':  round(final_ret, 2),
                    'actual_return_pct': round(final_ret, 2),
                    'predicted_return_pct': 0.0,
                    'exit_reason': 'END_OF_TEST',
                    'hold_days':   max(final_hold, 0),
                })
        equity_vals[-1] = capital  # reflejar posiciones cerradas en último punto

        # ── Per-asset statistics ───────────────────────────────────────────────
        per_asset_stats = {}
        for t in all_trades:
            sym = t['symbol']
            if sym not in per_asset_stats:
                per_asset_stats[sym] = {'trades': 0, 'pnl': 0.0, 'wins': 0}
            per_asset_stats[sym]['trades'] += 1
            per_asset_stats[sym]['pnl']    += t['pnl']
            if t['pnl'] > 0:
                per_asset_stats[sym]['wins'] += 1
        per_asset_list = sorted([
            {
                'symbol':   sym,
                'trades':   v['trades'],
                'pnl':      round(v['pnl'], 2),
                'win_rate': round(v['wins'] / v['trades'] * 100, 1) if v['trades'] > 0 else 0,
            }
            for sym, v in per_asset_stats.items()
        ], key=lambda x: x['pnl'], reverse=True)

        # Timeline de activos activos (muestreada a ~150 puntos)
        step = max(1, len(equity_dts) // 150)
        active_timeline = [
            {'date': equity_dts[i], 'symbols': active_syms_history.get(equity_dts[i], [])}
            for i in range(0, len(equity_dts), step)
        ]

        # ── 4. Métricas de rendimiento ────────────────────────────────────────
        final_val    = float(equity_vals[-1])
        total_ret    = (final_val - init_cap) / init_cap * 100
        eq_arr       = np.array(equity_vals, dtype=float)
        daily_rets   = np.diff(eq_arr) / (eq_arr[:-1] + 1e-8)
        sharpe       = (np.mean(daily_rets) / (np.std(daily_rets) + 1e-10)) * np.sqrt(252) if len(daily_rets) > 1 else 0.0
        # Sortino: penaliza solo la volatilidad bajista
        neg_rets     = daily_rets[daily_rets < 0]
        down_std     = float(np.std(neg_rets)) if len(neg_rets) > 1 else 1e-10
        sortino      = (np.mean(daily_rets) / (down_std + 1e-10)) * np.sqrt(252) if len(daily_rets) > 1 else 0.0
        running_max  = np.maximum.accumulate(eq_arr)
        drawdowns    = (eq_arr - running_max) / (running_max + 1e-8)
        max_dd       = float(drawdowns.min() * 100)
        # Calmar: retorno anualizado / max drawdown absoluto
        # Usar retorno total compuesto (no media aritmética) para evitar Calmar positivo
        # cuando el retorno total es negativo (bug: mean(daily_rets) puede ser >0 por asimetría)
        total_ret_dec = (eq_arr[-1] / eq_arr[0]) - 1 if eq_arr[0] > 0 else 0.0
        n_years       = max(len(daily_rets) / 252, 0.01)
        ann_ret       = float((1 + total_ret_dec) ** (1 / n_years) - 1)
        calmar        = ann_ret / max(abs(max_dd / 100), 0.001)
        winning      = [t for t in all_trades if t['pnl'] > 0]
        losing       = [t for t in all_trades if t['pnl'] <= 0]
        win_rate     = len(winning) / len(all_trades) * 100 if all_trades else 0
        avg_win      = float(np.mean([t['pnl'] for t in winning])) if winning else 0
        avg_loss     = float(np.mean([t['pnl'] for t in losing]))  if losing  else 0
        profit_factor = abs(avg_win * len(winning) / (avg_loss * len(losing) + 1e-8)) if losing else float('inf')
        avg_hold_days = float(np.mean([t['hold_days'] for t in all_trades])) if all_trades else 0.0

        # Buy-hold benchmark: igual-ponderado en los activos disponibles el primer día
        # Usamos los activos que tenían datos el primer día
        bh_vals   = []
        bh_syms   = [s for s, dm in asset_maps.items() if all_dates[0] in dm]
        if bh_syms:
            alloc = init_cap / len(bh_syms)
            for ds in all_dates:
                val = 0.0
                for s in bh_syms:
                    if s in asset_maps:
                        dates_avail = sorted(d for d in asset_maps[s] if d <= ds)
                        if dates_avail:
                            p0 = asset_maps[s][all_dates[0]]['close']
                            pt = asset_maps[s][dates_avail[-1]]['close']
                            val += alloc * pt / (p0 + 1e-10)
                bh_vals.append(round(val, 2))
        else:
            bh_vals = [init_cap] * len(all_dates)

        max_signal_val = float(max(all_ss)) if all_ss else 0.0
        bh_final_val   = float(bh_vals[-1]) if bh_vals else init_cap
        bh_ret_pct     = (bh_final_val - init_cap) / init_cap * 100

        return jsonify({
            'status': 'success',
            'watchlist': trained_ok,
            'period': {
                'test_start':  all_dates[0],
                'test_end':    all_dates[-1],
                'n_test_days': len(all_dates),
                'n_train_days': 0,
                'train_end':   start_date,
            },
            'config': {
                'initial_capital': init_cap,
                'signal_percentile': sig_pct,
                'kelly_scale': kelly_sc,
                'allow_short': allow_short,
                'mode': mode,
                'trend_reverse_exit': trend_reverse_exit,
                'disable_atr_stop': disable_atr_stop,
                'disable_adx_filter': disable_adx_filter,
                'max_holding_days': cfg.max_holding_days,
                'target_vol': cfg.target_vol,
                'use_llm': use_llm,
                'llm_provider': llm_provider if use_llm else None,
                'use_risk_gate': use_risk_gate,
            },
            'llm_stats': {
                'calls':           llm_stats['calls']           if use_llm else 0,
                'cache_hits':      llm_stats['cache_hits']      if use_llm else 0,
                'errors':          llm_stats['errors']          if use_llm else 0,
                'vetoed':          llm_stats['vetoed']          if use_llm else 0,
                'budget_skipped':  llm_stats['budget_skipped']  if use_llm else 0,
                'sent_real_news':  llm_stats.get('sent_real_news', 0)    if use_llm else 0,
                'sent_implied_only': llm_stats.get('sent_implied_only', 0) if use_llm else 0,
                'avg_latency':     round(llm_stats['total_time'] / max(llm_stats['calls'], 1), 2) if use_llm else 0.0,
            },
            'risk_gate_stats': risk_gate.stats() if risk_gate is not None else {'enabled': False},
            'filter_counts': filter_counts if debug_filter_counts else None,
            'performance': {
                'initial_capital':    init_cap,
                'final_value':        round(final_val, 2),
                'final_capital':      round(final_val, 2),   # alias para compatibilidad frontend
                'total_return_pct':   round(total_ret, 2),
                'buy_hold_return_pct': round(bh_ret_pct, 2),
                'sharpe_ratio':       round(float(sharpe), 3),
                'sortino_ratio':      round(float(sortino), 3),
                'calmar_ratio':       round(float(calmar), 3),
                'max_drawdown_pct':   round(max_dd, 2),
                'total_trades':       len(all_trades),
                'win_rate_pct':       round(win_rate, 1),
                'win_rate':           round(win_rate, 1),   # alias frontend
                'avg_win':            round(avg_win, 2),
                'avg_loss':           round(avg_loss, 2),
                'profit_factor':      round(float(profit_factor), 2) if profit_factor != float('inf') else 99.0,
                'avg_hold_days':      round(avg_hold_days, 1),
            },
            'equity_curve': {
                'dates':           equity_dts,
                'values':          [round(v, 2) for v in equity_vals],
                'portfolio_value': [round(v, 2) for v in equity_vals],  # alias frontend
                'buy_hold_value':  bh_vals,
            },
            'trades': all_trades[-100:],
            'per_asset': per_asset_list,
            'active_timeline': active_timeline,
            'signal_stats': {
                'total_signals':    len(all_ss),           # señales brutas generadas
                'filtered_signals': len(all_trades),       # señales que pasaron el umbral
                'threshold':        round(float(threshold), 6),
                'max_signal':       round(max_signal_val, 6),
                'filter_percentile': sig_pct,
            },
            'leakage_proof': True,
        })

    except ValueError as ve:
        return jsonify({'status': 'error', 'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Autonomous backtest error: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


# Keep old single-asset route for backwards compat (redirects to new autonomous endpoint)
@app.route('/api/paper/historical-backtest', methods=['POST'])
def paper_historical_backtest():
    return paper_autonomous_backtest()


# ── Multi-asset scanner ────────────────────────────────────────────────────────

# Watchlists por defecto
_STOCK_WATCHLIST  = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'META', 'SPY']
_CRYPTO_WATCHLIST = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD']


@app.route('/api/paper/scan', methods=['POST'])
def paper_scan():
    """
    POST /api/paper/scan
    Body: {
      asset_type: 'stock' | 'crypto' | 'both'  (default 'both')
      extra_symbols: ['AAPL', ...]  (opcional)
    }

    Escanea la watchlist de activos y devuelve señales rankeadas por fuerza.
    """
    body       = request.get_json(force=True) or {}
    atype_req  = body.get('asset_type', 'both')
    extra_syms = body.get('extra_symbols', [])

    watchlist = []
    if atype_req in ('stock', 'both'):
        watchlist += [{'symbol': s, 'asset_type': 'stock'} for s in _STOCK_WATCHLIST]
    if atype_req in ('crypto', 'both'):
        watchlist += [{'symbol': s, 'asset_type': 'crypto'} for s in _CRYPTO_WATCHLIST]
    for ex in extra_syms:
        sym = ex.get('symbol', '').strip().upper()
        if sym and not any(w['symbol'] == sym for w in watchlist):
            watchlist.append({'symbol': sym, 'asset_type': ex.get('asset_type', 'stock')})

    logger.info(f"=== Scanner: {len(watchlist)} activos ({atype_req}) ===")

    signals = []
    for item in watchlist:
        sym   = item['symbol']
        atype = item['asset_type']
        try:
            result = run_full_analysis(
                symbol=sym, timeframe='1y', asset_type=atype, use_llm=False
            )
            if result.get('status') != 'success':
                signals.append({'symbol': sym, 'asset_type': atype,
                                'action': 'ERROR', 'error': result.get('error', '?')})
                continue

            ml   = result.get('ml', {})
            pred = float(ml.get('ensemble_prediction', 0))
            conf = float(ml.get('ensemble_confidence', 0.5))
            ss   = abs(pred) * conf

            # Acción (alineado con BotConfig.min_signal_strength=0.001)
            if ss < 0.001:
                action = 'HOLD'
                reason = 'Señal débil'
            elif pred > 0:
                action = 'LONG'
                reason = f'Alcista +{pred*100:.2f}%, conf {conf:.0%}'
            else:
                action = 'SHORT'
                reason = f'Bajista {pred*100:.2f}%, conf {conf:.0%}'

            # Kelly sizing
            p = min(max(conf, 0.1), 0.9)
            b = max(abs(pred) / 0.01, 0.5)
            kelly = max(0.0, (p * b - (1 - p)) / b)
            suggested_size = round(min(kelly * 0.25 * 100, 30), 1)

            signals.append({
                'symbol':               sym,
                'asset_type':           atype,
                'action':               action,
                'predicted_return_pct': round(pred * 100, 3),
                'confidence':           round(conf, 3),
                'signal_strength':      round(ss, 5),
                'suggested_size_pct':   suggested_size,
                'current_price':        result['data_info'].get('current_price'),
                'reason':               reason,
                'already_open':         sym in _portfolio.positions,
            })

        except Exception as exc:
            logger.warning(f"Scanner error para {sym}: {exc}")
            signals.append({'symbol': sym, 'asset_type': atype,
                            'action': 'ERROR', 'error': str(exc)})

    # Rankear: primero LONG/SHORT (por fuerza desc), luego HOLD, luego ERROR
    def rank_key(s):
        if s['action'] in ('LONG', 'SHORT'):
            return (0, -s.get('signal_strength', 0))
        if s['action'] == 'HOLD':
            return (1, 0)
        return (2, 0)

    signals.sort(key=rank_key)

    return jsonify({
        'status':    'success',
        'timestamp': datetime.now().isoformat(),
        'scanned':   len(signals),
        'signals':   signals,
    })


# ══════════════════════════════════════════════════════════════════════════════
# PAPER TRADING ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

from trading_bot.paper_portfolio import PaperPortfolio


def _get_current_price(symbol: str, asset_type: str = 'stock') -> float:
    """
    Obtiene el precio actual con reintentos ante rate-limiting de Yahoo.
    Usa nuestra capa de descarga (data_loader) con backoff exponencial.
    """
    import time as _t
    # Intentar primero con yfinance (más rápido para precios intraday)
    for attempt in range(3):
        try:
            hist = yf.Ticker(symbol).history(period='2d')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as exc:
            err_str = str(exc).lower()
            if '429' in err_str or 'rate' in err_str or 'too many' in err_str:
                wait = (2 ** attempt) * 2
                logger.warning(f"Rate limited al obtener precio de {symbol}, esperando {wait}s")
                _t.sleep(wait)
            else:
                break  # error no recuperable
    # Fallback: usar nuestro data_loader con retry (descarga últimos 5 días)
    try:
        loader = FinancialDataLoader()
        today  = datetime.now().strftime('%Y-%m-%d')
        week   = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        df     = loader.download_data(symbol, week, today, asset_type=asset_type)
        if df is not None and not df.empty:
            return float(df['Close'].iloc[-1])
    except Exception:
        pass
    raise ValueError(f"No se pudo obtener precio actual de {symbol} tras reintentos")


# Instancia global de la cartera simulada (persiste entre peticiones)
_portfolio = PaperPortfolio()


@app.route('/api/paper/status', methods=['GET'])
def paper_status():
    """
    GET /api/paper/status?live=true

    Devuelve el estado completo de la cartera de paper trading.

    Query params:
        live (bool, default true): Si true, consulta precios actuales vía
                                   yfinance para calcular el P&L no realizado.
    """
    live = request.args.get('live', 'true').lower() not in ('false', '0', 'no')

    try:
        status = _portfolio.get_status(fetch_live_prices=live)
        return jsonify({'status': 'success', **status})

    except Exception as e:
        logger.error(f"Paper status error: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/paper/signal', methods=['POST'])
def paper_signal():
    """
    POST /api/paper/signal
    Body: { symbol, asset_type }

    Analiza un activo con el pipeline ML completo y devuelve la señal de
    trading sugerida junto con el sizing calculado mediante criterio de Kelly
    fraccionado.  No ejecuta ninguna operación.
    """
    body       = request.get_json(force=True)
    symbol     = body.get('symbol', '').strip().upper()
    asset_type = body.get('asset_type', 'stock')

    if not symbol:
        return jsonify({'status': 'error', 'error': 'symbol es obligatorio'}), 400

    try:
        # Análisis ML + multi-agente LLM completo
        result = run_full_analysis(symbol=symbol, timeframe='1y',
                                   asset_type=asset_type, use_llm=True)

        if result.get('status') != 'success':
            return jsonify(result)

        ml   = result.get('ml', {})
        pred = float(ml.get('ensemble_prediction', 0))
        conf = float(ml.get('ensemble_confidence', 0.5))
        signal_strength = abs(pred) * conf

        # Usar decisión híbrida ML + agentes
        hybrid = result.get('hybrid', {})
        agents = result.get('agents', {})

        # Acción: preferir recomendación híbrida si disponible
        hybrid_rec = hybrid.get('recommendation', '').upper()
        if hybrid_rec in ('COMPRAR', 'BUY'):
            action = 'LONG'
        elif hybrid_rec in ('VENDER', 'SELL'):
            action = 'SHORT'
        elif signal_strength < 0.001:
            action = 'HOLD'
        elif pred > 0:
            action = 'LONG'
        else:
            action = 'SHORT'

        # Reasoning del Portfolio Manager LLM
        pm_reasoning = agents.get('final_reasoning', '')
        if action == 'HOLD':
            reason = pm_reasoning or 'Señal demasiado débil para operar'
        elif action == 'LONG':
            reason = pm_reasoning or f'Predicción alcista ({pred * 100:+.2f}%), confianza {conf:.0%}'
        else:
            reason = pm_reasoning or f'Predicción bajista ({pred * 100:+.2f}%), confianza {conf:.0%}'

        # Sizing sugerido — Kelly fraccionado (máx. 30 % del capital)
        p = min(max(conf, 0.1), 0.9)
        b = max(abs(pred) / 0.01, 0.5)
        kelly = max(0, (p * b - (1 - p)) / b)
        suggested_size_pct = min(kelly * 0.25 * 100, 30)

        # Formatear análisis de cada agente para el frontend
        individual = agents.get('individual', {})
        agent_cards = {
            'technical': {
                'recommendation': individual.get('technical', {}).get('recommendation', '—'),
                'confidence':     individual.get('technical', {}).get('confidence', 0),
                'reasoning':      individual.get('technical', {}).get('reasoning', ''),
            },
            'risk': {
                'recommendation': individual.get('risk', {}).get('recommendation', '—'),
                'confidence':     individual.get('risk', {}).get('confidence', 0),
                'reasoning':      individual.get('risk', {}).get('reasoning', ''),
            },
            'sentiment': {
                'recommendation': individual.get('sentiment', {}).get('recommendation', '—'),
                'confidence':     individual.get('sentiment', {}).get('confidence', 0),
                'reasoning':      individual.get('sentiment', {}).get('reasoning', ''),
            },
            'ml': {
                'recommendation': 'ALCISTA' if pred > 0 else 'BAJISTA',
                'confidence':     round(conf, 3),
                'reasoning':      f'Predicción ML ensemble: {pred*100:+.3f}%',
            },
            'portfolio_manager': {
                'recommendation': agents.get('final_decision', '—'),
                'confidence':     agents.get('final_confidence', 0),
                'reasoning':      agents.get('final_reasoning', ''),
            },
        }

        return jsonify({
            'status':     'success',
            'symbol':     symbol,
            'asset_type': asset_type,
            'timestamp':  datetime.now().isoformat(),
            'action':                action,
            'predicted_return_pct':  round(pred * 100, 3),
            'confidence':            round(conf, 3),
            'signal_strength':       round(signal_strength, 5),
            'suggested_size_pct':    round(suggested_size_pct, 1),
            'current_price':         result['data_info'].get('current_price'),
            'reason':                reason,
            'agents':                agent_cards,
            'hybrid':                hybrid,
        })

    except Exception as e:
        logger.error(f"Paper signal error: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/paper/execute', methods=['POST'])
def paper_execute():
    """
    POST /api/paper/execute
    Body: { symbol, asset_type, action, size_pct }

    Ejecuta una operación en la cartera de paper trading.  Obtiene el precio
    actual de mercado vía yfinance y delega en PaperPortfolio.execute_trade().

    Campos del body:
        symbol     (str):   Ticker del activo (p.ej. 'AAPL').
        asset_type (str):   Tipo de activo ('stock', 'crypto', etc.).
        action     (str):   'LONG' o 'SHORT'.
        size_pct   (float): Fracción del efectivo a usar (0–100, en porcentaje).
    """
    body       = request.get_json(force=True)
    symbol     = body.get('symbol', '').strip().upper()
    asset_type = body.get('asset_type', 'stock')
    action     = body.get('action', '').strip().upper()
    size_pct   = float(body.get('size_pct', 10))  # viene en % (ej. 10 → 10%)

    if not symbol:
        return jsonify({'status': 'error', 'error': 'symbol es obligatorio'}), 400
    if not action:
        return jsonify({'status': 'error', 'error': 'action es obligatorio (LONG/SHORT)'}), 400

    try:
        # Obtener precio actual de mercado (con retry ante rate-limiting)
        try:
            current_price = _get_current_price(symbol, asset_type)
        except Exception as exc:
            logger.warning(f"[paper/execute] Error obteniendo precio para {symbol}: {exc}")
            return jsonify({'status': 'error',
                            'error': f'No se pudo obtener precio actual de {symbol}: {exc}'}), 400

        # Convertir size_pct de porcentaje (0-100) a fracción (0-1)
        size_fraction = size_pct / 100.0

        trade = _portfolio.execute_trade(
            symbol=symbol,
            asset_type=asset_type,
            action=action,
            size_pct=size_fraction,
            current_price=current_price,
            signal_strength=float(body.get('signal_strength', 0.0)),
            predicted_return_pct=float(body.get('predicted_return_pct', 0.0)),
            reason=body.get('reason', 'MANUAL'),
        )

        status = _portfolio.get_status(fetch_live_prices=False)

        return jsonify({
            'status':    'success',
            'trade':     trade,
            'portfolio': status,
        })

    except ValueError as e:
        return jsonify({'status': 'error', 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Paper execute error: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/paper/close', methods=['POST'])
def paper_close():
    """
    POST /api/paper/close
    Body: { symbol }

    Cierra una posición abierta en la cartera de paper trading al precio
    actual de mercado.
    """
    body       = request.get_json(force=True)
    symbol     = body.get('symbol', '').strip().upper()
    # Detectar tipo de activo: los criptos terminan en -USD o lo indica explícitamente
    asset_type = body.get('asset_type',
                          'crypto' if ('-USD' in symbol or symbol in ('BTC','ETH','SOL','BNB','XRP'))
                          else 'stock')

    if not symbol:
        return jsonify({'status': 'error', 'error': 'symbol es obligatorio'}), 400

    try:
        # Precio actual de cierre (con retry ante rate-limiting)
        try:
            current_price = _get_current_price(symbol, asset_type)
        except Exception as exc:
            logger.warning(f"[paper/close] Error obteniendo precio para {symbol}: {exc}")
            return jsonify({'status': 'error',
                            'error': f'No se pudo obtener precio actual de {symbol}: {exc}'}), 400

        trade  = _portfolio.close_position(symbol, current_price=current_price,
                                           reason=body.get('reason', 'MANUAL'))
        status = _portfolio.get_status(fetch_live_prices=False)

        return jsonify({
            'status':    'success',
            'trade':     trade,
            'portfolio': status,
        })

    except KeyError as e:
        return jsonify({'status': 'error', 'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Paper close error: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/paper/reset', methods=['POST'])
def paper_reset():
    """
    POST /api/paper/reset
    Body: { initial_capital } (opcional, por defecto 100 000)

    Cierra todas las posiciones abiertas y reinicia la cartera al capital
    inicial indicado.
    """
    body            = request.get_json(force=True) or {}
    initial_capital = body.get('initial_capital', 100_000.0)

    try:
        status = _portfolio.reset(initial_capital=float(initial_capital))
        return jsonify({'status': 'success', 'portfolio': status})

    except Exception as e:
        logger.error(f"Paper reset error: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/paper/sessions', methods=['GET'])
def paper_sessions_list():
    """GET /api/paper/sessions — lista todas las sesiones guardadas."""
    try:
        sessions = _portfolio.list_sessions()
        return jsonify({'status': 'success', 'sessions': sessions})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/paper/sessions/save', methods=['POST'])
def paper_sessions_save():
    """POST /api/paper/sessions/save — guarda la sesión actual con un nombre."""
    body = request.get_json(force=True) or {}
    name = body.get('name', '').strip()
    desc = body.get('description', '')
    if not name:
        return jsonify({'status': 'error', 'error': 'name es obligatorio'}), 400
    try:
        info = _portfolio.save_session(name, desc)
        return jsonify({'status': 'success', **info})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/paper/sessions/load', methods=['POST'])
def paper_sessions_load():
    """POST /api/paper/sessions/load — carga una sesión guardada."""
    body = request.get_json(force=True) or {}
    name = body.get('name', '').strip()
    if not name:
        return jsonify({'status': 'error', 'error': 'name es obligatorio'}), 400
    try:
        status = _portfolio.load_session(name)
        return jsonify({'status': 'success', 'portfolio': status})
    except FileNotFoundError as e:
        return jsonify({'status': 'error', 'error': str(e)}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/paper/sessions/delete', methods=['POST'])
def paper_sessions_delete():
    """POST /api/paper/sessions/delete — elimina una sesión guardada."""
    body = request.get_json(force=True) or {}
    name = body.get('name', '').strip()
    if not name:
        return jsonify({'status': 'error', 'error': 'name es obligatorio'}), 400
    deleted = _portfolio.delete_session(name)
    return jsonify({'status': 'success', 'deleted': deleted})


@app.route('/api/paper/live-analysis', methods=['GET'])
def paper_live_analysis():
    """
    GET /api/paper/live-analysis
    Para cada posición abierta ejecuta un análisis ML rápido (sin LLM para
    velocidad) y devuelve la señal actualizada + razonamiento del agente técnico.
    """
    try:
        status = _portfolio.get_status(fetch_live_prices=True)
        positions = status.get('positions_with_pnl', [])

        analysis = []
        for pos in positions:
            sym   = pos['symbol']
            atype = pos.get('asset_type', 'stock')
            try:
                result = run_full_analysis(sym, timeframe='1y', asset_type=atype, use_llm=False)
                ml  = result.get('ml', {})
                pred = float(ml.get('ensemble_prediction', 0))
                conf = float(ml.get('ensemble_confidence', 0.5))
                agents_r = result.get('agents', {})
                tech_r   = agents_r.get('individual', {}).get('technical', {})
                risk_r   = agents_r.get('individual', {}).get('risk', {})
                analysis.append({
                    'symbol':           sym,
                    'current_price':    pos['current_price'],
                    'unrealized_pnl':   pos['unrealized_pnl'],
                    'unrealized_pnl_pct': pos['unrealized_pnl_pct'],
                    'ml_pred_pct':      round(pred * 100, 3),
                    'ml_confidence':    round(conf, 3),
                    'ml_direction':     'ALCISTA' if pred > 0 else 'BAJISTA',
                    'tech_rec':         tech_r.get('recommendation', '—'),
                    'tech_reasoning':   tech_r.get('reasoning', ''),
                    'risk_rec':         risk_r.get('recommendation', '—'),
                    'risk_reasoning':   risk_r.get('reasoning', ''),
                    'should_close':     (
                        abs(pred) > 0.005 and (
                            (pos['action'] == 'LONG'  and pred < -0.003) or
                            (pos['action'] == 'SHORT' and pred >  0.003)
                        )
                    ),
                })
            except Exception as exc:
                analysis.append({'symbol': sym, 'error': str(exc)})

        return jsonify({
            'status':    'success',
            'timestamp': datetime.now().isoformat(),
            'portfolio': status,
            'analysis':  analysis,
        })
    except Exception as e:
        logger.error(f"Live analysis error: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ──────────────────────────────────────────────
if __name__ == '__main__':
    _port = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 60)
    print("  FINANCIAL AI - ML + LLM Hybrid Prediction System")
    print(f"  Open http://localhost:{_port} in your browser")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=_port, debug=False)
