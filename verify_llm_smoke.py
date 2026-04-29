"""
Smoke test mínimo del proveedor LLM local.

Prueba sólo el camino LLM aislado — sin pandas/yfinance/ML/agents.
Objetivo: demostrar que la ruta `LLMClient(provider=...)` funciona en esta
máquina sin depender de memoria adicional del resto de la pipeline.

Uso:
    .venv/bin/python -u verify_llm_smoke.py                    # default 'local' (Qwen2.5-0.5B bfloat16)
    TFG_LLM_PROVIDER=local-gguf .venv/bin/python -u verify_llm_smoke.py   # Tier 3.4 (Qwen2.5-1.5B Q4_K_M)
"""
import os, sys, time, resource
import builtins as _b
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import warnings; warnings.filterwarnings('ignore')

_orig = _b.print
def _p(*a, **k): k.setdefault('flush', True); _orig(*a, **k)
_b.print = _p

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.llm_client import LLMClient

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

PROVIDER = os.environ.get('TFG_LLM_PROVIDER', 'local')

print("=" * 78)
print(f"SMOKE TEST — LLMClient(provider='{PROVIDER}')")
print("=" * 78)
print(f"RSS inicial: {rss_mb():.1f} MB")

cli = LLMClient(provider=PROVIDER)
print(f"Model id:    {cli.model}")

# Inferencia 1: sentiment analysis
news = (
    "Microsoft Q3 earnings beat estimates: revenue up 17% YoY driven by Azure. "
    "Cloud margins expand. Analysts raise price targets after strong AI guidance."
)
print(f"\n[1/2] analyze_sentiment — input: '{news[:80]}...'")
t0 = time.time()
r1 = cli.analyze_sentiment(news)
print(f"      took {time.time()-t0:.1f}s  |  RSS tras carga+gen: {rss_mb():.1f} MB")
parsed = r1.get('parsed', {}) if isinstance(r1, dict) else {}
print(f"      sentiment:  {parsed.get('sentiment','?')}")
print(f"      score:      {parsed.get('score','?')}")
print(f"      confidence: {parsed.get('confidence','?')}")
reasoning = str(parsed.get('reasoning', ''))[:200]
print(f"      reasoning:  {reasoning}")

# Inferencia 2: market interpretation (el segundo camino LLM)
print(f"\n[2/2] interpret_market_data — usa cache (debería ser rápido en carga)")
t0 = time.time()
r2 = cli.interpret_market_data(
    technical_summary="RSI=62 (neutral-alto), MACD cruzando al alza, SMA20 > SMA50, volumen +20% vs media",
    sentiment_summary="Sentimiento agregado: BULLISH (score +0.7, confianza 0.8). Noticias positivas de earnings.",
    risk_summary="Volatilidad 22% anualizada (percentil 45), drawdown máx reciente 6%. Riesgo: MEDIO.",
    ml_summary="ML ensemble: retorno esperado +4.2% a 5 días, confianza 0.9, acuerdo entre modelos 75%.",
)
print(f"      took {time.time()-t0:.1f}s  |  RSS actual: {rss_mb():.1f} MB")
parsed2 = r2.get('parsed', {}) if isinstance(r2, dict) else {}
print(f"      recommendation: {parsed2.get('recommendation','?')}")
print(f"      confidence:     {parsed2.get('confidence','?')}")
print(f"      position_size:  {parsed2.get('position_size','?')}")
print(f"      risk_level:     {parsed2.get('risk_level','?')}")
reasoning2 = str(parsed2.get('reasoning', ''))[:240]
print(f"      reasoning:      {reasoning2}")

print("\n" + "=" * 78)
print("OK · LLM local verificado end-to-end (ambos caminos)")
print("=" * 78)
