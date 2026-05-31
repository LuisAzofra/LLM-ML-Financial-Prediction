"""
finbert_sentiment.py — Sentimiento financiero con FinBERT (ProsusAI/finbert).

FinBERT es un modelo Transformer (BERT) FINETUNEADO sobre texto financiero
(Financial PhraseBank, Malo et al. 2014). A diferencia de VADER o un LLM
generalista, está especializado en distinguir el tono alcista/bajista de
titulares y reportes financieros, que es exactamente el propósito de la
capa de sentimiento de este TFG.

Se integra en dos puntos del sistema:
  · utils/llm_client.LLMClient.analyze_sentiment — sustituye el parser por
    palabras clave por el clasificador especializado cuando está disponible.
  · api.py modo index_trend (flag `use_finbert_filter`) — filtro DEFENSIVO de
    re-entrada: no recompra el índice si el flujo de noticias reciente es
    fuertemente bajista.

Diseño causal/honesto (clave para no falsear el backtest):
  NewsFetcher sirve RSS del MOMENTO ACTUAL — no existe archivo histórico
  point-in-time de noticias. Por tanto, para fechas pasadas la función
  `index_news_sentiment` devuelve NEUTRAL e inactivo: así el backtest
  histórico NO sufre look-ahead (no usa noticias de hoy para decidir en 2018)
  y reproduce EXACTAMENTE la curva del baseline Faber-QQQ+parking. El filtro
  solo se activa en ventanas recientes (≤ `recency_days`) y en trading en vivo.

Degradación elegante: si transformers/torch o el modelo no están disponibles,
todas las funciones devuelven neutral y el sistema sigue funcionando con su
comportamiento previo.
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("FINBERT_MODEL", "ProsusAI/finbert")

_PIPELINE = None          # singleton perezoso
_LOAD_FAILED = False      # si falla una vez, no reintentar en cada llamada
_SCORE_CACHE: Dict[str, Dict[str, float]] = {}


def _get_pipeline():
    """Carga perezosa del pipeline de FinBERT (una sola vez por proceso)."""
    global _PIPELINE, _LOAD_FAILED
    if _PIPELINE is not None or _LOAD_FAILED:
        return _PIPELINE
    if os.environ.get("FINBERT_DISABLE", "0") == "1":
        _LOAD_FAILED = True
        return None
    try:
        from transformers import pipeline
        _PIPELINE = pipeline("text-classification", model=MODEL_NAME, top_k=None)
        logger.info(f"FinBERT cargado: {MODEL_NAME}")
    except Exception as e:
        _LOAD_FAILED = True
        logger.info(f"FinBERT no disponible ({e}); se usa fallback neutral")
        _PIPELINE = None
    return _PIPELINE


def is_available() -> bool:
    return _get_pipeline() is not None


def score_text(text: str) -> Dict[str, float]:
    """Puntúa un texto. Devuelve probas y un score neto en [-1, 1].

    net = P(positive) - P(negative). label = clase de mayor probabilidad.
    Texto vacío o modelo ausente → neutral.
    """
    text = (text or "").strip()
    if not text:
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "net": 0.0, "label": "neutral"}
    if text in _SCORE_CACHE:
        return _SCORE_CACHE[text]
    pipe = _get_pipeline()
    if pipe is None:
        out = {"pos": 0.0, "neg": 0.0, "neu": 1.0, "net": 0.0, "label": "neutral"}
        _SCORE_CACHE[text] = out
        return out
    try:
        # FinBERT tiene un límite de 512 tokens; recortamos por caracteres.
        scores = pipe(text[:1000])[0]
        probs = {d["label"].lower(): float(d["score"]) for d in scores}
        pos, neg, neu = probs.get("positive", 0.0), probs.get("negative", 0.0), probs.get("neutral", 0.0)
        net = pos - neg
        label = max(("positive", pos), ("negative", neg), ("neutral", neu), key=lambda x: x[1])[0]
        out = {"pos": pos, "neg": neg, "neu": neu, "net": net, "label": label}
    except Exception as e:
        logger.debug(f"FinBERT score error: {e}")
        out = {"pos": 0.0, "neg": 0.0, "neu": 1.0, "net": 0.0, "label": "neutral"}
    _SCORE_CACHE[text] = out
    return out


def score_headlines(headlines: List[str]) -> Dict[str, Any]:
    """Agrega el sentimiento de una lista de titulares.

    net = media de los netos individuales. Útil para resumir el flujo de
    noticias de un símbolo en un solo número accionable.
    """
    headlines = [h for h in (headlines or []) if h and h.strip()]
    if not headlines:
        return {"net": 0.0, "label": "neutral", "n": 0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
    rows = [score_text(h) for h in headlines]
    n = len(rows)
    net = sum(r["net"] for r in rows) / n
    pos = sum(r["pos"] for r in rows) / n
    neg = sum(r["neg"] for r in rows) / n
    neu = sum(r["neu"] for r in rows) / n
    label = "positive" if net > 0.15 else "negative" if net < -0.15 else "neutral"
    return {"net": net, "label": label, "n": n, "pos": pos, "neg": neg, "neu": neu}


def to_sentiment_parsed(agg: Dict[str, Any]) -> Dict[str, Any]:
    """Mapea el agregado de FinBERT al esquema que espera analyze_sentiment."""
    net = float(agg.get("net", 0.0))
    label_map = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}
    sentiment = label_map.get(agg.get("label", "neutral"), "neutral")
    # Confianza = fuerza de la señal (|net|), acotada y conservadora.
    confidence = float(min(0.2 + abs(net) * 0.8, 0.95)) if agg.get("n", 0) > 0 else 0.2
    return {
        "sentiment": sentiment,
        "score": round(net, 4),
        "confidence": round(confidence, 4),
        "key_points": [],
        "reasoning": f"FinBERT sobre {agg.get('n', 0)} titulares (net={net:+.3f})",
        "time_sensitivity": "short_term",
        "source": "finbert",
    }


def index_news_sentiment(
    symbol: str,
    as_of_date: Optional[str] = None,
    recency_days: int = 7,
    max_items: int = 20,
    is_crypto: bool = False,
) -> Dict[str, Any]:
    """Sentimiento del flujo de noticias de un símbolo, para el filtro de re-entrada.

    CAUSAL: si `as_of_date` es histórica (más de `recency_days` antes de hoy),
    devuelve inactivo+neutral — no hay archivo point-in-time, así que usar el
    RSS actual sería look-ahead. Solo se activa en fechas recientes / en vivo.
    """
    today = datetime.now().date()
    if as_of_date:
        try:
            d = datetime.strptime(str(as_of_date)[:10], "%Y-%m-%d").date()
            if (today - d).days > recency_days:
                return {"net": 0.0, "label": "neutral", "n_real": 0, "active": False}
        except Exception:
            pass
    if not is_available():
        return {"net": 0.0, "label": "neutral", "n_real": 0, "active": False}
    try:
        from data.news_fetcher import get_news_for_symbol
        news = get_news_for_symbol(symbol, is_crypto=is_crypto, max_items=max_items)
        titles = [n.get("title", "") for n in news if n.get("title")]
        agg = score_headlines(titles)
        agg["n_real"] = agg.pop("n", 0)
        agg["active"] = agg["n_real"] > 0
        return agg
    except Exception as e:
        logger.debug(f"index_news_sentiment error {symbol}: {e}")
        return {"net": 0.0, "label": "neutral", "n_real": 0, "active": False}
