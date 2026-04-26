"""
CLI: análisis ML+LLM como subproceso, imprime JSON a stdout.

Dos modos según env:
- TFG_LIGHTWEIGHT=0 (default) → pipeline COMPLETO (run_full_analysis):
    4 modelos ML (RF + XGB + LGB + HGB) con CV walk-forward, Optuna,
    GARCH, multi-agente (Technical + Sentiment + Risk + PortfolioManager),
    todos con LLM real. Tarda más (~60-180 s) y consume más RAM (~2-3 GB
    pico) pero es el comportamiento real de la app.
- TFG_LIGHTWEIGHT=1 → modo ligero autosuficiente (sin importar api.py):
    sólo yfinance + indicadores + ML proxy momentum + 2 llamadas LLM.
    Tarda 20-30 s y pico ~1 GB.

Aislar el LLM en un subproceso permite que el server Flask se mantenga en
~70 MB y que cada análisis sea independiente — al morir el subproceso
liberamos toda la memoria.

Uso:
    python -u cli_analyze.py <symbol> <timeframe> <asset_type> [provider]
"""
import os, sys, json, traceback, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import numpy as np

from data.data_loader import FinancialDataLoader
from utils.data_utils import DataProcessor
from utils.llm_client import LLMClient


TIMEFRAME_DAYS = {
    '1m': 30, '3m': 90, '6m': 180, '1y': 365, '2y': 730, '3y': 1095,
    '5M': 30, '1H': 30, '12H': 90, '24H': 365, '1W': 365, '1M': 730, '1Y': 1095,
}


def analyze(symbol: str, timeframe: str, asset_type: str, provider: str) -> dict:
    days = TIMEFRAME_DAYS.get(timeframe, 365)
    end   = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    out = {'symbol': symbol, 'asset_type': asset_type, 'timeframe': timeframe,
           'start_date': start, 'end_date': end, 'mode': 'lightweight_subprocess',
           'status': 'running'}

    loader = FinancialDataLoader()
    df = loader.download_data(symbol, start, end, asset_type=asset_type)
    if df is None or df.empty:
        out['status'] = 'error'; out['error'] = f'No data for {symbol}'
        return out

    actual_symbol = df['Symbol'].iloc[0] if 'Symbol' in df.columns else symbol
    dp = DataProcessor()
    df = dp.clean_data(df)
    df = dp.add_technical_indicators(df)
    last = df.iloc[-1]

    out['data_info'] = {
        'records': len(df),
        'period_start': str(df.index[0].date()),
        'period_end':   str(df.index[-1].date()),
        'current_price': round(float(last['Close']), 4),
        'annualized_volatility': round(float(last.get('Volatility', 0) * 100), 2)
            if 'Volatility' in df.columns else None,
    }
    chart_df = df.tail(min(len(df), 500))
    out['price_chart'] = {
        'dates': [str(d.date()) for d in chart_df.index],
        'close': [round(float(v), 4) for v in chart_df['Close'].values],
        'sma_20': [round(float(v), 4) if not np.isnan(v) else None
                   for v in chart_df['SMA_20'].values] if 'SMA_20' in chart_df.columns else [],
        'sma_50': [round(float(v), 4) if not np.isnan(v) else None
                   for v in chart_df['SMA_50'].values] if 'SMA_50' in chart_df.columns else [],
    }
    out['garch'] = None

    # ── ML proxy ─────────────────────────────────────────────────────
    close = df['Close']
    r5  = float(close.pct_change(5).iloc[-1])  if len(close) > 5  else 0.0
    r20 = float(close.pct_change(20).iloc[-1]) if len(close) > 20 else 0.0
    pred_return = 0.5 * r5 + 0.5 * r20
    direction = 'UP' if pred_return > 0 else 'DOWN'
    ml_conf  = round(0.5 + min(abs(pred_return) * 10, 0.4), 3)
    agreement = round(0.7 + min(abs(pred_return) * 5, 0.25), 3)
    out['ml'] = {
        'ensemble_prediction': round(pred_return, 6),
        'ensemble_confidence': ml_conf,
        'ensemble_direction':  direction,
        'model_agreement':     agreement,
        'best_model':          'lightweight_proxy',
        'horizon_days':        5,
        'mode':                'proxy_momentum',
    }

    # ── Resúmenes ────────────────────────────────────────────────────
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

    # ── LLM (carga única en este subproceso) ─────────────────────────
    llm = LLMClient(provider=provider)
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
    if '|' in raw_rec:
        raw_rec = 'HOLD'
    llm_conf = float(mkt_p.get('confidence', 0.5) or 0.5)

    REC_MAP = {'BUY': 'COMPRAR', 'SELL': 'VENDER', 'HOLD': 'MANTENER',
               'WAIT': 'ESPERAR', 'STRONG_BUY': 'COMPRAR_FUERTE', 'STRONG_SELL': 'VENDER_FUERTE'}
    final_dec = REC_MAP.get(raw_rec, raw_rec)

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
    out['agents'] = {
        'final_decision':   final_dec,
        'final_confidence': round(llm_conf, 3),
        'final_reasoning':  str(mkt_p.get('reasoning', ''))[:600],
        'individual':       individual,
    }

    # ── Hybrid score ─────────────────────────────────────────────────
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
    out['hybrid'] = {
        'score': round(hybrid_score, 4),
        'confidence': round(hybrid_conf, 3),
        'recommendation': hybrid_rec,
        'direction_conflict': bool(disagree),
        'ml_signal':  round(ml_signal, 3),
        'llm_signal': round(llm_signal, 3),
    }
    out['status'] = 'success'
    out['summary'] = {
        'recommendation': hybrid_rec,
        'hybrid_score':   out['hybrid']['score'],
        'confidence':     out['hybrid']['confidence'],
        'ml_direction':   direction,
        'agent_decision': final_dec,
    }
    return out


def analyze_full(symbol: str, timeframe: str, asset_type: str, provider: str) -> dict:
    """Pipeline completo: ML ensemble + GARCH + multi-agente + LLM."""
    # Import perezoso: sólo si vamos al modo completo cargamos api.py
    # (que arrastra xgboost, lightgbm, arch, agentes, etc.).
    from api import run_full_analysis
    result = run_full_analysis(
        symbol=symbol, timeframe=timeframe,
        asset_type=asset_type, use_llm=True, llm_provider=provider,
    )
    if isinstance(result, dict):
        result.setdefault('mode', 'full_subprocess')
    return result


def main():
    if len(sys.argv) < 3:
        sys.stdout.write(json.dumps({'status': 'error', 'error': 'usage: cli_analyze.py SYM TF AT [PROV]'}))
        sys.exit(2)
    symbol     = sys.argv[1]
    timeframe  = sys.argv[2] if len(sys.argv) > 2 else '6m'
    asset_type = sys.argv[3] if len(sys.argv) > 3 else 'stock'
    provider   = sys.argv[4] if len(sys.argv) > 4 else os.environ.get('TFG_LLM_PROVIDER', 'local')
    use_light  = os.environ.get('TFG_LIGHTWEIGHT', '0') == '1'
    try:
        if use_light:
            result = analyze(symbol, timeframe, asset_type, provider)
        else:
            result = analyze_full(symbol, timeframe, asset_type, provider)
    except Exception as e:
        result = {'status': 'error', 'symbol': symbol, 'error': str(e),
                  'traceback': traceback.format_exc()}
    sys.stdout.write(json.dumps(result, default=str))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
