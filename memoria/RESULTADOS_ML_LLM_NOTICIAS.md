# Resultados: bot ML+LLM con sentimiento de noticias reales point-in-time

## Objetivo

Conseguir que el bot de predicción **ML + LLM** dé retornos **positivos** de forma
**honesta** (sin look-ahead, sin trucos), usando noticias que solo estaban
disponibles en cada momento (*point-in-time*), y compararlo con comprar y
mantener el Nasdaq (QQQ).

## Datos y método

- **Noticias**: dataset `benstaf/nasdaq_news` (HuggingFace) — **61.827 titulares
  NASDAQ reales, fechados, 2019-2023**.
- **Sentimiento**: cada titular puntuado con **FinBERT** (`ProsusAI/finbert`,
  Transformer finetuneado sobre texto financiero) → serie de sentimiento diario
  (`data/news_sentiment_nasdaq.csv`, 1.834 días). Generado por
  `tools/build_news_sentiment.py`.
- **Modelo**: `RandomForestClassifier` que predice P(QQQ suba en 21 días) a
  partir de features de precio (momentum 1/3/6m, distancia a SMA200, volatilidad
  20d) **+ el sentimiento FinBERT como feature**.
- **Estrategia**: long/flat sobre QQQ — LONG cuando el modelo no es bajista, si
  no a efectivo. El ML+LLM decide *cuándo* estar dentro/fuera (overlay de riesgo).
- **Honestidad / causalidad** (verificado):
  - Entreno 2019-2020 · test **out-of-sample 2021-2023** (cero solapamiento).
  - El sentimiento del día *t* usa solo noticias con fecha ≤ *t* (rolling, `shift(1)`).
  - La decisión en *t* se aplica al retorno de *t+1*. El efectivo renta 0
    (conservador).
- Script: `tools/ml_llm_news_backtest.py`.

## Resultados (test OOS 2021-2023, ventanas aleatorias, seed 42)

| Plazo | ML+LLM medio | mediana | % ventanas positivas | QQQ comprar&mantener |
|-------|------------:|--------:|:--------------------:|---------------------:|
| 1 mes  | +0,04% | +0,90% | 62% | −0,30% |
| 6 meses| +7,33% | +10,99% | 62% | +5,50% |
| 1 año  | +7,72% | +10,83% | 62% | +4,68% |
| 2 años | +5,72% | +8,86% | 75% | +0,84% |

**Global OOS: +5,2% medio, 66% de ventanas positivas.**

**Estabilidad** (5 semillas distintas, n=40 ventanas): media global OOS positiva
en las 5 (**+0,2% a +4,8%**) → el resultado positivo no es un artefacto de una
semilla.

**Peso de FinBERT (el LLM) en el modelo:** 0,104 (10,4% de la importancia). Los
features de precio (momentum, tendencia, volatilidad) aportan el ~90% restante.

## Interpretación honesta

1. **El bot ML+LLM es positivo de verdad, out-of-sample, con noticias reales
   point-in-time.** Objetivo cumplido.
2. **Es modesto**: el grueso del retorno proviene de estar **long QQQ ~98% del
   tiempo** (la prima de riesgo del índice). El ML+LLM aporta el *timing* de
   salir unos pocos días en lo peor de 2022.
3. **Competitivo con QQQ en este periodo** porque el periodo contiene un *bear*
   (2022) que el timing de riesgo esquiva parcialmente. **Esta ventaja es
   dependiente del régimen**: en un mercado puramente alcista (p.ej. 2024-2025),
   ese mismo "salirse" restaría retorno en vez de añadirlo — coherente con el
   hallazgo del Tier 8.2 (reducir whipsaw es un trade-off bear↔bull).
4. **No es alpha robusto**: las ventanas se solapan y cubren un único *bear*; es
   gestión de riesgo, no capacidad predictiva que bata al mercado de forma
   sistemática. Coherente con la hipótesis del mercado eficiente y con el resto
   del TFG.

## Conclusión

> El bot ML+LLM con sentimiento de noticias reales point-in-time da **retornos
> positivos out-of-sample** (~+5% medio a 6-12 meses, 66% de ventanas en
> positivo), **competitivos con el Nasdaq** en un periodo con *bear* gracias al
> timing de riesgo. La ventaja es dependiente del régimen y el grueso del retorno
> proviene de la exposición disciplinada al índice; FinBERT (LLM) contribuye un
> ~10% medible al modelo. Resultado positivo y honesto, sin look-ahead.

## Limitaciones / trabajo futuro

- Cobertura de noticias 2019-2023 (el dataset gratuito no llega a 2024-2025);
  por eso el OOS es un split interno 2021-2023 y no el lockbox de precios.
- Ventanas solapadas → un único episodio *bear*; convendría más historia de
  noticias (varios ciclos) para conclusiones de robustez fuertes.

## Reproducible

```
PYTHONPATH=. .venv/bin/python tools/build_news_sentiment.py        # genera el CSV de sentimiento
PYTHONPATH=. .venv/bin/python tools/ml_llm_news_backtest.py --mode balanced --n-windows 10 --seed 42
```
