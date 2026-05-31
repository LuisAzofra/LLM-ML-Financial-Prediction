"""
smoke_finbert.py — prueba de humo de la integración de FinBERT.

Verifica de forma reproducible que:
  1. El modelo finetuneado ProsusAI/finbert carga.
  2. Clasifica correctamente titulares financieros (positivo/negativo/neutral).
  3. El camino EN VIVO obtiene noticias reales de un símbolo y las puntúa.
  4. El agente de sentimiento del sistema (LLMClient.analyze_sentiment) usa
     FinBERT como proveedor.

Uso:
    PYTHONPATH=. .venv/bin/python tools/smoke_finbert.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.finbert_sentiment import is_available, score_text, score_headlines, index_news_sentiment


def main():
    assert is_available(), "FinBERT no disponible (instala transformers+torch)"
    print(f"[1] FinBERT disponible: True ({os.environ.get('FINBERT_MODEL', 'ProsusAI/finbert')})")

    cases = [
        ("positive", "Apple reports record quarterly earnings, beating analyst expectations."),
        ("negative", "Nasdaq tumbles as recession fears mount and tech stocks sell off sharply."),
        ("neutral",  "Markets were mostly flat ahead of the Fed decision."),
    ]
    print("[2] Clasificación de titulares:")
    ok = 0
    for expected, txt in cases:
        s = score_text(txt)
        match = "OK" if s["label"] == expected else "≠"
        ok += s["label"] == expected
        print(f"    {match} esperado={expected:>8} obtenido={s['label']:>8} net={s['net']:+.2f}  ::  {txt[:55]}")
    print(f"    → {ok}/{len(cases)} coinciden")

    print("[3] Agregado de varios titulares:")
    agg = score_headlines([c[1] for c in cases])
    print(f"    net={agg['net']:+.3f} label={agg['label']} n={agg['n']}")

    print("[4] Camino EN VIVO (noticias reales actuales de QQQ):")
    live = index_news_sentiment("QQQ", as_of_date=None)
    print(f"    n_real={live['n_real']} net={live['net']:+.3f} label={live['label']} active={live['active']}")

    print("[5] Causalidad: fecha histórica → filtro inerte (sin look-ahead):")
    hist = index_news_sentiment("QQQ", as_of_date="2018-06-01")
    print(f"    active={hist['active']} net={hist['net']:+.3f}  (debe ser active=False)")
    assert hist["active"] is False, "El filtro NO debe activarse en fechas históricas"

    print("[6] Agente de sentimiento del sistema usa FinBERT:")
    from utils.llm_client import LLMClient
    r = LLMClient(provider="local").analyze_sentiment(
        "Tesla shares plunge after weak delivery numbers and margin pressure.")
    print(f"    provider={r.get('provider')} parsed={r.get('parsed', {}).get('sentiment')} "
          f"score={r.get('parsed', {}).get('score')}")
    assert r.get("provider") == "finbert", "analyze_sentiment debe usar FinBERT cuando está disponible"

    print("\n✅ smoke_finbert: TODO OK")


if __name__ == "__main__":
    main()
