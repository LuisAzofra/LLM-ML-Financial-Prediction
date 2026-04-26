"""
debate.py — Tier 2.3: bull/bear debate + judge LLM.

Inspirado en HKUDS/Vibe-Trading investment_committee y AI-Trader paper.
Hipótesis: el LLM-único en single-pass es ruidoso (Qwen 0.5B especialmente).
Forzar al modelo a articular caso bull AISLADO + caso bear AISLADO + un
judge neutral mejora la calibración de confianza, sin reemplazar la
dirección que decide el ML (preserva ML como decisor primario).

Diseño:
  Pasada 1 (bull): "argumenta STRONG BUY asumiendo tesis alcista" → 3 puntos.
  Pasada 2 (bear): "argumenta STRONG SELL asumiendo tesis bajista" → 3 puntos.
  Pasada 3 (judge): toma {bull, bear, ml_pred, risk, sentiment} → confidence ∈ [0,1].
    El judge SÓLO modula sizing (multiplicador 0.3-1.0 sobre Kelly), NO flippea
    la dirección de ML.

Coste: 3× llamadas LLM por decisión (vs 1 single-pass). Cache trimestral
del backtest amortiza esto: ~12 trimestres × N símbolos vs 730 días.

API pública:
- `run_debate(llm, ml_pred_str, technical, sentiment, risk)` → dict con
  bull_args, bear_args, confidence_mult, ok flag.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_bull_prompt(ml_pred: str, technical: str, sentiment: str, risk: str) -> str:
    return (
        "You are a strict BULLISH equity/crypto analyst. Argue the strongest LONG case.\n\n"
        f"TECHNICAL: {technical}\n"
        f"SENTIMENT: {sentiment}\n"
        f"RISK: {risk}\n"
        f"ML PREDICTION: {ml_pred}\n\n"
        "Provide exactly 3 concrete bullet points supporting a LONG entry.\n"
        "Be terse, factual. No hedging.\n"
        "Return ONLY a single JSON object on one line:\n"
        '{"bull_args": ["point 1", "point 2", "point 3"]}'
    )


def build_bear_prompt(ml_pred: str, technical: str, sentiment: str, risk: str) -> str:
    return (
        "You are a strict BEARISH equity/crypto analyst. Argue against the LONG entry.\n\n"
        f"TECHNICAL: {technical}\n"
        f"SENTIMENT: {sentiment}\n"
        f"RISK: {risk}\n"
        f"ML PREDICTION: {ml_pred}\n\n"
        "Provide exactly 3 concrete bullet points warning AGAINST entering LONG.\n"
        "Be terse, factual. No hedging.\n"
        "Return ONLY a single JSON object on one line:\n"
        '{"bear_args": ["risk 1", "risk 2", "risk 3"]}'
    )


def build_judge_prompt(
    ml_pred: str, bull_args: List[str], bear_args: List[str],
    sentiment: str, risk: str,
) -> str:
    bull_block = "\n".join(f"  - {a}" for a in bull_args[:3]) if bull_args else "  (no args)"
    bear_block = "\n".join(f"  - {a}" for a in bear_args[:3]) if bear_args else "  (no args)"
    return (
        "You are a NEUTRAL JUDGE evaluating a trade decision.\n\n"
        "ML decides DIRECTION; your role is to output a SIZING multiplier ∈ [0.3, 1.0].\n\n"
        f"ML PREDICTION: {ml_pred}\n"
        f"SENTIMENT: {sentiment}\n"
        f"RISK: {risk}\n\n"
        "BULL CASE:\n" + bull_block + "\n\n"
        "BEAR CASE:\n" + bear_block + "\n\n"
        "Output exactly ONE number for `confidence`:\n"
        "  1.0 → bull case clearly dominates (full size)\n"
        "  0.7 → bull case stronger, bear has merit (70% size)\n"
        "  0.5 → balanced, both sides valid (50% size)\n"
        "  0.3 → bear case dominates (30% size)\n\n"
        "Do NOT flip direction. Only modulate sizing.\n"
        "Return ONLY a single JSON object on one line:\n"
        '{"confidence": 0.X}'
    )


def _safe_parse_json(text: str, key: str, default) -> Any:
    """Extrae el primer JSON object con la key requerida; si no, devuelve default."""
    if not text:
        return default
    # Buscar el primer { ... } balanceado
    start = text.find('{')
    if start < 0:
        return default
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                blob = text[start:i + 1]
                try:
                    obj = json.loads(blob)
                    if isinstance(obj, dict) and key in obj:
                        return obj[key]
                except Exception:
                    pass
                # Continúa buscando otra ocurrencia
                start = text.find('{', i + 1)
                if start < 0:
                    return default
                depth = 0
    return default


def run_debate(
    llm,
    ml_pred_str: str,
    technical: str,
    sentiment: str,
    risk: str,
    max_tokens_args: int = 400,
    max_tokens_judge: int = 150,
) -> Dict[str, Any]:
    """
    Ejecuta el flujo bull → bear → judge en `llm` (LLMClient).

    Returns dict con:
      - bull_args:        List[str] (≤3)
      - bear_args:        List[str] (≤3)
      - confidence_mult:  float ∈ [0.3, 1.0] (multiplier sobre Kelly)
      - ok:               True si las 3 pasadas devolvieron contenido
      - raw:              {bull_raw, bear_raw, judge_raw} para debug
    """
    out = {
        'bull_args':       [],
        'bear_args':       [],
        'confidence_mult': 0.5,  # default: balanced si algo falla
        'ok':              False,
        'raw':             {},
    }

    # Pasada 1: bull
    try:
        r_bull = llm.generate(
            build_bull_prompt(ml_pred_str, technical, sentiment, risk),
            system_prompt="You output ONLY valid JSON. No markdown, no prose outside JSON.",
            temperature=0.3, max_tokens=max_tokens_args,
        )
        bull_text = r_bull.get('text', '') if isinstance(r_bull, dict) else str(r_bull)
        out['raw']['bull'] = bull_text[:500]
        bull_args = _safe_parse_json(bull_text, 'bull_args', [])
        if isinstance(bull_args, list):
            out['bull_args'] = [str(a)[:200] for a in bull_args[:3]]
    except Exception as e:
        logger.debug(f"debate bull failed: {e}")
        out['raw']['bull_err'] = str(e)[:200]

    # Pasada 2: bear
    try:
        r_bear = llm.generate(
            build_bear_prompt(ml_pred_str, technical, sentiment, risk),
            system_prompt="You output ONLY valid JSON. No markdown, no prose outside JSON.",
            temperature=0.3, max_tokens=max_tokens_args,
        )
        bear_text = r_bear.get('text', '') if isinstance(r_bear, dict) else str(r_bear)
        out['raw']['bear'] = bear_text[:500]
        bear_args = _safe_parse_json(bear_text, 'bear_args', [])
        if isinstance(bear_args, list):
            out['bear_args'] = [str(a)[:200] for a in bear_args[:3]]
    except Exception as e:
        logger.debug(f"debate bear failed: {e}")
        out['raw']['bear_err'] = str(e)[:200]

    # Pasada 3: judge — sólo si tenemos al menos UNO de los lados
    if not out['bull_args'] and not out['bear_args']:
        return out  # ok=False; caller debe usar fallback

    try:
        r_judge = llm.generate(
            build_judge_prompt(ml_pred_str, out['bull_args'], out['bear_args'], sentiment, risk),
            system_prompt="You output ONLY valid JSON. No markdown, no prose outside JSON.",
            temperature=0.2, max_tokens=max_tokens_judge,
        )
        judge_text = r_judge.get('text', '') if isinstance(r_judge, dict) else str(r_judge)
        out['raw']['judge'] = judge_text[:500]
        conf = _safe_parse_json(judge_text, 'confidence', 0.5)
        try:
            cf = float(conf)
        except Exception:
            cf = 0.5
        cf = max(0.3, min(1.0, cf))
        out['confidence_mult'] = cf
        out['ok'] = True
    except Exception as e:
        logger.debug(f"debate judge failed: {e}")
        out['raw']['judge_err'] = str(e)[:200]
    return out


def is_veto_from_debate(direc: str, confidence_mult: float, threshold: float = 0.4) -> bool:
    """
    El judge NO flippea dirección; si su confidence_mult cae por debajo del
    threshold, el caller puede tratar la operación como vetada (HOLD).
    """
    return confidence_mult < threshold
