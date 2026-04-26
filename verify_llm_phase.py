"""
Fase 2 aislada: análisis multi-agente con LLM local (Qwen2.5-0.5B).

Esta versión evita el OOM que producía `run_full_analysis` al entrenar los
modelos ML (XGBoost + LightGBM + Optuna 25 trials) justo antes de cargar el
LLM de ~1 GB. Aquí:

1. Descargamos datos una vez (ligero).
2. Construimos un ``ml_pred`` sintético pero realista (direction + confianza
   razonables) en lugar de entrenar modelos.
3. Llamamos directamente a ``_run_agents``, que es exactamente la ruta LLM
   que queríamos validar end-to-end: TechnicalAnalyst, SentimentAnalyst,
   RiskManager y PortfolioManager ↓ Qwen2.5-0.5B local.
4. Calculamos también el hybrid score para verificar la combinación ML+LLM.

Ejecutar:
    TFG_LLM_PROVIDER=local .venv/bin/python verify_llm_phase.py
"""
import os
import sys
import warnings
import builtins as _b
warnings.filterwarnings('ignore')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Fuerza stdout line-buffered: si el proceso muere (OOM) no perdemos prints.
_orig_print = _b.print
def _print(*a, **kw):
    kw.setdefault('flush', True)
    _orig_print(*a, **kw)
_b.print = _print

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import FinancialDataLoader
from utils.data_utils import DataProcessor
import api as _api  # reutilizamos _run_agents y _compute_hybrid_score

LLM_PROVIDER = os.environ.get('TFG_LLM_PROVIDER', 'local')
SYMBOLS = [('MSFT', 'stock'), ('BTC-USD', 'crypto')]
YEARS = 1


def _build_synthetic_ml_pred(df):
    """
    Construye un ml_pred realista a partir de la serie de cierre.
    Usa el retorno reciente y su volatilidad para derivar una dirección y
    confianza plausibles — suficiente para alimentar a los agentes.
    """
    close = df['Close'].dropna()
    # retorno medio 5d + tendencia 20d normalizados
    r5  = float(close.pct_change(5).iloc[-1]) if len(close) > 5 else 0.0
    r20 = float(close.pct_change(20).iloc[-1]) if len(close) > 20 else 0.0
    pred_return = 0.6 * r5 + 0.4 * r20  # proxy de predicción ensemble
    direction = 'UP' if pred_return > 0 else ('DOWN' if pred_return < 0 else 'FLAT')
    # confianza basada en magnitud — escalada a [0.5, 0.9]
    conf = 0.5 + min(abs(pred_return) * 10.0, 0.4)
    return {
        'ensemble_prediction': float(pred_return),
        'ensemble_confidence': float(conf),
        'ensemble_direction':  direction,
        'model_agreement':     0.75,          # simulado
        'best_model':          'ensemble_synthetic',
        'horizon_days':        5,
    }


def _run_one(symbol: str, asset_type: str):
    print(f"\n>>> {symbol} ({asset_type})")
    loader = FinancialDataLoader()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=YEARS * 365)).strftime('%Y-%m-%d')

    df = loader.download_data(symbol, start_date, end_date, asset_type=asset_type)
    if df is None or df.empty:
        print(f"  datos insuficientes — saltando")
        return

    dp = DataProcessor()
    df = dp.clean_data(df)
    df = dp.add_technical_indicators(df)

    ml_pred = _build_synthetic_ml_pred(df)
    print(f"  ML (sintético): ret={ml_pred['ensemble_prediction']*100:+.2f}%   "
          f"dir={ml_pred['ensemble_direction']}   conf={ml_pred['ensemble_confidence']:.2f}")

    # Llamada directa a los agentes con LLM local — evita el OOM.
    agt = _api._run_agents(df, symbol, ml_pred, use_llm=True, llm_provider=LLM_PROVIDER)

    print(f"  Agentes (LLM):  decision={agt.get('final_decision','?')}   "
          f"conf={agt.get('final_confidence','?')}")
    for name, info in (agt.get('individual') or {}).items():
        r_txt = str(info.get('reasoning', ''))[:200]
        print(f"    · {name}: {info.get('recommendation','?')} "
              f"(conf {info.get('confidence','?')})")
        if r_txt:
            print(f"      ↳ {r_txt}")

    # Hybrid score (ML 75% / LLM 25%) — usa ml_pred sintético como ml_result.
    hy = _api._compute_hybrid_score(ml_pred, agt)
    print(f"  Híbrido:        reco={hy.get('recommendation','?')}   "
          f"score={hy.get('score','?')}   conf={hy.get('confidence','?')}")


def main():
    print("=" * 78)
    print(f"FASE 2 — ANÁLISIS MULTI-AGENTE CON LLM (provider={LLM_PROVIDER})")
    print("=" * 78)
    print("NOTA: ML sintético para evitar OOM (ML real se valida en verify_improvements.py)")

    for sym, atype in SYMBOLS:
        try:
            _run_one(sym, atype)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FALLO {sym}: {e}")


if __name__ == '__main__':
    main()
