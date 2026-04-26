"""
Demo end-to-end del bot ML+LLM en acción.

Muestra el flujo real que hace `run_full_analysis` pero más ligero (sin Optuna
ni GARCH ni xgboost) para que entre en RAM en esta máquina:

  1. Descarga precios del símbolo (yfinance).
  2. Calcula indicadores técnicos (RSI, SMA, MACD, ATR) — DataProcessor.
  3. Genera predicción ML (proxy: retorno reciente + momentum) → ml_pred.
  4. Llama al LLM local Qwen2.5-0.5B para:
       · analyze_sentiment(noticias del símbolo)
       · interpret_market_data(técnico, sentimiento, riesgo, ML)
  5. Combina los 2 caminos en un *hybrid score* (mismo cálculo que el sistema).
  6. Imprime la DECISIÓN del bot: BUY / SELL / HOLD + razonamiento del LLM.

Uso:
    .venv/bin/python -u demo_bot_llm.py [SÍMBOLO]
    .venv/bin/python -u demo_bot_llm.py MSFT
    .venv/bin/python -u demo_bot_llm.py BTC-USD
"""
import os, sys, time
import builtins as _b
import warnings; warnings.filterwarnings('ignore')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

_orig_print = _b.print
def _print(*a, **k): k.setdefault('flush', True); _orig_print(*a, **k)
_b.print = _print

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from utils.llm_client import LLMClient
from data.data_loader import FinancialDataLoader
from utils.data_utils import DataProcessor


SYMBOL    = (sys.argv[1] if len(sys.argv) > 1 else 'MSFT').upper()
ASSET_TYPE = 'crypto' if SYMBOL.endswith('-USD') else 'stock'
MONTHS    = 6  # ventana corta — más ligero en memoria


# ─────────────────────────────────────────────────────────────────
# 1. Datos
# ─────────────────────────────────────────────────────────────────
print("=" * 78)
print(f"DEMO BOT ML+LLM — {SYMBOL} ({ASSET_TYPE}) · ventana {MONTHS} meses")
print("=" * 78)

loader = FinancialDataLoader()
end   = datetime.now().strftime('%Y-%m-%d')
start = (datetime.now() - timedelta(days=MONTHS * 31)).strftime('%Y-%m-%d')
print(f"\n[1/5] Descargando datos {SYMBOL} {start}→{end}…")
df = loader.download_data(SYMBOL, start, end, asset_type=ASSET_TYPE)
if df is None or df.empty or len(df) < 50:
    print(f"  ❌ Sin datos suficientes")
    sys.exit(1)

dp = DataProcessor()
df = dp.clean_data(df)
df = dp.add_technical_indicators(df)
last = df.iloc[-1]
print(f"  ✓ {len(df)} filas · precio actual: {float(last['Close']):,.2f}")


# ─────────────────────────────────────────────────────────────────
# 2. Predicción ML (proxy ligero)
# ─────────────────────────────────────────────────────────────────
print(f"\n[2/5] Predicción ML (proxy momentum + recent returns)")
close = df['Close']
r5  = float(close.pct_change(5).iloc[-1])
r20 = float(close.pct_change(20).iloc[-1])
pred_return = 0.5 * r5 + 0.5 * r20
direction = 'UP' if pred_return > 0 else 'DOWN'
ml_conf = 0.5 + min(abs(pred_return) * 10, 0.4)
agreement = 0.7 + min(abs(pred_return) * 5, 0.25)
print(f"  · retorno 5d: {r5*100:+.2f}%   retorno 20d: {r20*100:+.2f}%")
print(f"  · ml_pred:    return={pred_return*100:+.2f}%  dir={direction}  conf={ml_conf:.2f}  agreement={agreement:.2f}")


# ─────────────────────────────────────────────────────────────────
# 3. Resúmenes para los prompts del LLM
# ─────────────────────────────────────────────────────────────────
print(f"\n[3/5] Construyendo resúmenes (técnico/sentimiento/riesgo/ML)")
rsi = float(last.get('RSI', 50)) if 'RSI' in df.columns else 50
sma20 = float(last.get('SMA_20', 0)) if 'SMA_20' in df.columns else 0
sma50 = float(last.get('SMA_50', 0)) if 'SMA_50' in df.columns else 0
sma200 = float(last.get('SMA_200', 0)) if 'SMA_200' in df.columns else 0
macd = float(last.get('MACD', 0)) if 'MACD' in df.columns else 0
macd_sig = float(last.get('MACD_signal', 0)) if 'MACD_signal' in df.columns else 0
volat = float(last.get('Volatility', 0) * 100) if 'Volatility' in df.columns else 0
price = float(last['Close'])

trend = []
if sma20 and sma50: trend.append('SMA20>SMA50' if sma20 > sma50 else 'SMA20<SMA50')
if sma50 and sma200: trend.append('SMA50>SMA200' if sma50 > sma200 else 'SMA50<SMA200')
trend_txt = ', '.join(trend) or 'n/a'

technical_summary = (
    f"Precio actual: ${price:,.2f}. RSI: {rsi:.1f} "
    f"({'sobreventa' if rsi<30 else 'sobrecompra' if rsi>70 else 'neutral'}). "
    f"MACD: {macd:.4f} vs signal {macd_sig:.4f} "
    f"({'cruzando al alza' if macd > macd_sig else 'bajista'}). "
    f"Tendencia: {trend_txt}. Volatilidad anualizada: {volat:.1f}%."
)
print(f"  técnico: {technical_summary}")

risk_summary = (
    f"Volatilidad {volat:.1f}% anualizada. "
    f"Riesgo: {'ALTO' if volat>40 else 'MEDIO' if volat>20 else 'BAJO'}. "
    f"Recomendación tamaño: {'pequeño' if volat>40 else 'medio'}."
)
print(f"  riesgo:  {risk_summary}")

ml_summary = (
    f"ML ensemble: retorno esperado a 5d = {pred_return*100:+.2f}%, "
    f"confianza {ml_conf:.2f}, acuerdo entre modelos {agreement:.2f}, "
    f"dirección {direction}."
)
print(f"  ML:      {ml_summary}")


# ─────────────────────────────────────────────────────────────────
# 4. LLM Qwen2.5-0.5B (local) — sentimiento + decisión
# ─────────────────────────────────────────────────────────────────
print(f"\n[4/5] Inferencia LLM local (Qwen2.5-0.5B bfloat16)")
llm = LLMClient(provider='local')

# Texto de noticias plausibles (sin tirar de RSS para evitar más memoria)
news = (
    f"{SYMBOL} cotiza a ${price:,.2f}. "
    f"Tendencia técnica {trend_txt}. "
    f"RSI {rsi:.0f}. Volatilidad reciente {volat:.0f}%. "
    f"Retorno últimos 20 días {r20*100:+.1f}%."
)

t0 = time.time()
print(f"  → analyze_sentiment …")
sent = llm.analyze_sentiment(news)
sent_p = sent.get('parsed', {}) if isinstance(sent, dict) else {}
sent_score = float(sent_p.get('score', 0) or 0)
sent_conf  = float(sent_p.get('confidence', 0.5) or 0.5)
sent_label = sent_p.get('sentiment', 'neutral')
print(f"    ✓ ({time.time()-t0:.1f}s) sentiment={sent_label}  score={sent_score:+.2f}  conf={sent_conf:.2f}")
print(f"    razonamiento: {str(sent_p.get('reasoning',''))[:200]}")

sentiment_summary = (
    f"Sentimiento agregado: {sent_label.upper()} (score {sent_score:+.2f}, conf {sent_conf:.2f}). "
    f"{str(sent_p.get('reasoning',''))[:160]}"
)

t0 = time.time()
print(f"\n  → interpret_market_data …")
mkt = llm.interpret_market_data(
    technical_summary=technical_summary,
    sentiment_summary=sentiment_summary,
    risk_summary=risk_summary,
    ml_summary=ml_summary,
)
mkt_p = mkt.get('parsed', {}) if isinstance(mkt, dict) else {}
print(f"    ✓ ({time.time()-t0:.1f}s)")
print(f"    recommendation: {mkt_p.get('recommendation','?')}")
print(f"    confidence:     {mkt_p.get('confidence','?')}")
print(f"    position_size:  {mkt_p.get('position_size','?')}")
print(f"    risk_level:     {mkt_p.get('risk_level','?')}")
print(f"    razonamiento:   {str(mkt_p.get('reasoning',''))[:280]}")


# ─────────────────────────────────────────────────────────────────
# 5. Hybrid score → decisión final del bot
# ─────────────────────────────────────────────────────────────────
print(f"\n[5/5] Combinación HÍBRIDA ML (75%) + LLM (25%) → decisión bot")

ml_signal = max(-1.0, min(1.0, pred_return * 100))
DECISION_MAP = {
    'COMPRAR': 0.7, 'COMPRA': 0.7, 'BUY': 0.7,
    'COMPRA_FUERTE': 1.0, 'STRONG_BUY': 1.0,
    'VENDER': -0.7, 'VENTA': -0.7, 'SELL': -0.7,
    'VENTA_FUERTE': -1.0, 'STRONG_SELL': -1.0,
    'MANTENER': 0.0, 'HOLD': 0.0, 'WAIT': 0.0, 'ESPERAR': 0.0,
}
llm_decision = str(mkt_p.get('recommendation', 'HOLD')).upper().strip()
# Si el modelo devolvió un placeholder tipo "BUY|SELL|HOLD" → tratamos como HOLD
if '|' in llm_decision:
    llm_decision = 'HOLD'
llm_signal = DECISION_MAP.get(llm_decision, 0.0)
llm_conf = float(mkt_p.get('confidence', 0.5) or 0.5)

hybrid_score = 0.75 * ml_signal * ml_conf + 0.25 * llm_signal * llm_conf
hybrid_conf  = 0.75 * ml_conf + 0.25 * llm_conf

# Direction-consistency: si ML y LLM discrepan → cap a HOLD
disagree = (ml_signal > 0 and llm_signal < 0) or (ml_signal < 0 and llm_signal > 0)
if disagree:
    final = 'MANTENER (discrepancia ML↔LLM)'
elif hybrid_score > 0.4:
    final = 'COMPRAR'
elif hybrid_score < -0.4:
    final = 'VENDER'
else:
    final = 'MANTENER'

print(f"  ML signal:     {ml_signal:+.3f}  · ML conf:  {ml_conf:.2f}")
print(f"  LLM signal:    {llm_signal:+.3f}  · LLM conf: {llm_conf:.2f}  ({llm_decision})")
print(f"  Hybrid score:  {hybrid_score:+.3f}")
print(f"  Hybrid conf:   {hybrid_conf:.2f}")

print("\n" + "─" * 78)
print(f"  🤖 BOT DECISION ({SYMBOL}):  {final}")
print(f"     justificación LLM: {str(mkt_p.get('reasoning',''))[:260]}")
print("─" * 78)
