# Resultados y conclusiones honestas

Este documento recoge la evaluación rigurosa del sistema tras corregir los sesgos
metodológicos detectados, y la conclusión honesta sobre su rentabilidad real.

## 1. Correcciones metodológicas aplicadas

Antes de medir nada, se corrigieron errores que inflaban artificialmente los
resultados. Sin estas correcciones, cualquier cifra de rentabilidad es ruido.

- **Data leakage en el escalado de modelos**: el scaler se ajustaba sobre
  train+test antes del split temporal. Corregido: se ajusta solo con train
  (+ embargo en el target de volatilidad).
- **Look-ahead en la ingesta**: `bfill` rellenaba huecos con precios futuros e
  interpolación bidireccional. Corregido a relleno solo hacia delante.
- **Umbral de señal no causal**: el percentil de señal se calculaba sobre toda
  la distribución del periodo de test. Corregido a ventana expanding (causal).
- **Curva de equity con calendario mixto**: con activos de calendario distinto
  (cripto cotiza findes, acciones no) las posiciones desaparecían del valor de
  cartera en los días sin barra, generando saltos fantasma de ±50% que inflaban
  Sharpe/Sortino y el max drawdown. Corregido con forward-fill del último cierre.
  Efecto: la volatilidad diaria de la curva bajó de ~25% a ~1.7% (realista).
- **Backtest con señales aleatorias**: se eliminó un método de backtest que
  generaba señales con `np.random.choice` (placeholder con look-ahead).
- **Ratio de Sortino mal definido**: usaba `std` de los retornos negativos.
  Corregido a la downside deviation estándar.
- **Benchmark injusto**: el buy&hold de referencia era el propio universo (con
  cripto ×10). Se añadieron benchmarks realistas: SPY y cartera 60/40.

## 2. Protocolo de evaluación

Para no sesgar por una sola ventana favorable, se evalúa en una **cuadrícula
multi-régimen**: 10 fechas de inicio (2018–2023, cubriendo pre-COVID, crash
COVID, recuperación, bull 2021, bear 2022 y recuperación 2023) × 4 plazos de
tenencia (3M, 6M, 1A, 2A). Cada variante se mide en ~40 backtests independientes.

También se realizó un **test de sesgo de supervivencia**: repetir la evaluación
sobre un universo amplio y aleatorio del S&P 500 (snapshot histórico) en lugar de
las 6-10 acciones "famosas" elegidas a posteriori.

## 3. Resultados — rendimiento por plazo de tenencia

Variante principal (`aggr_plus`: trend-following multi-activo con rotación por
momentum). Se reportan **media** y **mediana** porque divergen mucho:

| Plazo | Mediana (caso típico) | Media (inflada por cola) | % ventanas en positivo | SPY mediana |
|-------|----------------------:|-------------------------:|-----------------------:|------------:|
| 3 meses | −8.7% | −13.0% | 30% | +3.1% |
| 6 meses | −6.6% | +4.9% | 40% | +11.8% |
| 1 año | +5.4% | +29.9% | 60% | +15.7% |
| 2 años | +25.6% | +132.9% | 80% | +32.8% |

**Por qué media ≠ mediana.** Las 10 pruebas a 2 años, ordenadas, fueron:
`−26%, −22%, +12%, +20%, +22%, +29%, +147%, +239%, +371%, +537%`.
La media (+132.9%) la disparan 3-4 aciertos de cripto en 2020-21; **6 de las 10
pruebas quedaron por debajo de la media**. La mediana (+25.6%) refleja el
resultado típico. Para juzgar utilidad debe usarse la mediana.

## 4. ¿Es rentable? Comparación honesta vs comprar un índice

- **En términos absolutos**: sí a plazos largos. A 2 años el bot acaba en
  positivo el 80% de las veces (+25.6% típico). A corto plazo (≤6 meses)
  típicamente pierde.
- **Frente a comprar y mantener SPY**: no compensa. En el caso típico el SPY
  rinde igual o más (a 2 años SPY +32.8% vs bot +25.6%), y el bot solo supera al
  SPY en el **42%** de las ventanas — un cara o cruz — asumiendo mucho más riesgo.
- **Ajustado a riesgo (Sharpe)**: SPY 0.97 vs bot 0.25. El SPY es ~4× mejor por
  unidad de riesgo.

## 5. Palancas de mejora probadas (y por qué no mejoran)

Se probaron de forma rigurosa, midiendo cada una en la cuadrícula completa:

| Palanca | Resultado | Sharpe medio |
|---------|-----------|-------------:|
| Baseline (aggr_plus) | referencia | 0.25 |
| Universo diversificado (bonos/oro/materias primas) | peor | −0.48 |
| Señal TSMOM multi-lookback | peor | −0.38 |
| Sizing por riesgo (inverse-vol + vol-target) | menor drawdown, menor retorno | 0.16 |
| Holdings más largos (252/504/sin tope) | peor (devuelve ganancias en bear) | ≤0.07 |

**Ninguna mejora la rentabilidad ajustada a riesgo del baseline.** Además, el
propio baseline está probablemente sobreajustado: el **Deflated Sharpe Ratio**
(Bailey & López de Prado) indica que, con el número de configuraciones probadas
(~50-100), el Sharpe esperado por puro azar es 0.82–0.92 — por encima del 0.25
observado. Seguir buscando configuraciones que "ganen" en estas ventanas sería
overfitting, no una mejora real.

El tradeoff riesgo-rentabilidad es ineludible: el sizing por riesgo recorta el
drawdown (−27.9% → −23.9% a 2 años) pero baja la media (+132.9% → +71.3%), porque
la media alta procede precisamente de la exposición volátil (cripto) que el
control de riesgo reduce. No existe configuración que suba el retorno y baje el
riesgo a la vez.

## 6. Test de sesgo de supervivencia

Sobre un universo amplio y aleatorio del S&P 500 (en lugar de las acciones
elegidas a posteriori), la estrategia **pierde dinero a todos los plazos**
(retorno medio ≈ −11%, supera al buy&hold solo el ~8% de las veces). Esto
confirma que buena parte del rendimiento aparente del universo reducido era
selección de activos en retrospectiva, no habilidad del sistema.

## 7. Conclusión honesta

El sistema **gana dinero en términos absolutos a plazos largos**, pero **no bate
al indexado pasivo ajustando por riesgo**, y en un universo realista sin sesgo de
supervivencia no es rentable. Su media elevada depende de aciertos de cola
(cripto) no repetibles a voluntad.

Esto es coherente con la hipótesis del mercado eficiente y con la literatura: un
sistema técnico sobre datos diarios no obtiene una ventaja estructural sostenible
en estos activos. **No es un fracaso del trabajo, sino un resultado válido y
defendible.**

El valor del proyecto reside en:
1. Un **sistema completo y funcional** end-to-end (datos reales → ML + GARCH →
   multi-agente con LLM → gestión de riesgo → backtesting → interfaz).
2. Una **metodología de evaluación rigurosa**: sin look-ahead, con test de sesgo
   de supervivencia, benchmarks justos y conciencia de overfitting (Deflated
   Sharpe / PBO).
3. Un **hallazgo honesto**: la sofisticación (ML + LLM + multi-agente) no se
   traduce en alpha sobre el indexado pasivo en riesgo-ajustado.

## 8. Limitaciones y trabajo futuro

- Datos diarios de Yahoo Finance; sin datos intradía ni alternativos.
- Universo limitado; el test de supervivencia ya muestra la fragilidad fuera del
  conjunto reducido.
- Validación futura recomendada: **lockbox temporal** (reservar 2024+ como datos
  nunca vistos) y **Combinatorial Purged CV** para estimar la probabilidad de
  overfitting (PBO) de cualquier mejora antes de darla por buena.
- Posible reorientación del sistema hacia herramienta de **gestión de riesgo**
  (capturar subida con menor drawdown) o de **análisis/decisión** (el motor
  multi-agente + LLM como apoyo), donde su utilidad es más defendible que como
  generador de alpha.

## 9. Tier 8 — mejoras honestas sobre Faber-QQQ, validadas fuera de muestra

Partiendo de la mejor configuración honesta (Faber-QQQ: mantener QQQ cuando está
sobre su SMA200, si no efectivo) se realizó el trabajo futuro de la sección 8: se
endureció el protocolo (muchas fechas **aleatorias con semilla** por horizonte,
2 semanas a 2 años; universos **DESARROLLO 2014-2023** y **LOCKBOX 2024-2025**
disjuntos en el tiempo; alpha frente a SPY/QQQ/60-40; caché de precios causal;
**test de causalidad** automático y **Deflated Sharpe**) y se buscaron mejoras
reales validadas en el lockbox una sola vez.

**S1 — Parking en letras del Tesoro (ACEPTADO).** El baseline dejaba el efectivo
al 0% mientras estaba fuera de mercado (~20-30% del tiempo). Acreditar el tipo
libre de riesgo (`^IRX`, con retardo de un día, causal) sobre ese efectivo es
contabilidad correcta, no una apuesta. Mejora estricta y significativa sobre
Faber-QQQ en todos los horizontes, en desarrollo y en lockbox (Δretorno
+0.36 / +0.68 / +1.12 pp a 6M / 1A / 2A en lockbox, IC95% excluye el 0,
**180/180 ventanas nunca empeoran**, drawdown no aumenta). Mayor en lockbox por
los tipos al 4-5% de 2024-2025. Sin parámetro ajustado ⇒ sin riesgo de overfitting.

**Resultado destacado vs benchmark (lockbox 2024-2025, out-of-sample).**
Faber-QQQ+parking **bate al 60/40 de forma robusta**: alpha +7.99 pp a 1 año y
**+19.34 pp a 2 años, ganándole al 60/40 el 100% de las ventanas**, con drawdown
medio ~−13% (frente a la caída de un QQQ puro). Frente al SPY **empata o supera
ligeramente** (alpha +0.61 / +0.78 pp a 1A / 2A) con menos drawdown. Matiz
honesto: contra el 60/40 esto es sobre todo **beta de crecimiento + filtro de
caídas**, no alpha; y **sigue sin batir a su propio subyacente (QQQ) en un bull**.

**S2/S3/S4 — reducción de whipsaw (RECHAZADAS como mejora de retorno).** Banda de
histéresis, rebalanceo mensual (Faber canónico) y voto multi-lookback no superan
al baseline con parking en media, y en el lockbox (bull puro) lo penalizan
(el mensual −12.84 pp a 2A). El desglose por régimen es honesto e informativo: el
**mensual es un filtro de caídas mucho mejor** — en ventanas con QQQ a la baja
pierde −1.16% frente al −6.13% de QQQ y **le gana el 81% de las veces** (vs 34%
del Faber diario, que se desangra en whipsaw) — pero retrasa ~1 pp en los bull.
Queda como **palanca de gestión de riesgo**, no de retorno.

**Insight estructural (honesto).** Una estrategia *long/flat* sobre un único
índice no puede batir a comprar-y-mantenerlo en un bull sostenido; su valor es
esquivar drawdown en los bear. El lockbox 2024-2025, al ser bull puro, no tiene
mercado bajista que premie esa protección. Coherente con la sección 7: el valor
defendible de este enfoque es **gestión de riesgo** (exposición a crecimiento con
filtro de caídas y menos drawdown que el índice), no la generación de alpha.

## 10. FinBERT y bot ML+LLM con noticias reales point-in-time

Tras constatar que la capa de noticias no podía evaluarse sobre el pasado con el
lector RSS (que solo sirve titulares del momento actual; usar noticias de hoy para
decidir en 2018 sería look-ahead), se incorporó una base de datos real de noticias
fechadas para que el LLM participe en la decisión de forma honesta.

**FinBERT (LLM finetuneado financiero).** Se integró `ProsusAI/finbert` (BERT
afinado sobre Financial PhraseBank) como proveedor del agente de sentimiento
(`utils/finbert_sentiment.py`, `utils/llm_client.py`) y como filtro defensivo de
re-entrada en el modo `index_trend` (causal: inerte en histórico, activo en vivo).

**Base de datos de noticias point-in-time.** Dataset `benstaf/nasdaq_news`
(HuggingFace): **61.827 titulares NASDAQ fechados 2019-2023**. FinBERT puntúa cada
titular → serie de sentimiento diario (`data/news_sentiment_nasdaq.csv`, 1.834
días; `tools/build_news_sentiment.py`).

**Estrategia ML+LLM honesta** (`tools/ml_llm_news_backtest.py`). RandomForest
(momentum 1/3/6m, distancia a SMA200, volatilidad 20d **+ sentimiento FinBERT**)
decide long/flat sobre QQQ. Entreno 2019-2020, test **out-of-sample 2021-2023**;
sentimiento del día *t* limitado a noticias con fecha ≤ *t* (`shift(1)`); decisión
en *t* aplicada al retorno de *t+1*; efectivo renta 0 (conservador). No-look-ahead
verificado.

| Plazo | ML+LLM medio | mediana | % ventanas positivas | QQQ comprar&mantener |
|-------|------------:|--------:|:--------------------:|---------------------:|
| 1 mes  | +0,04% | +0,90% | 62% | −0,30% |
| 6 meses| +7,33% | +10,99% | 62% | +5,50% |
| 1 año  | +7,72% | +10,83% | 62% | +4,68% |
| 2 años | +5,72% | +8,86% | 75% | +0,84% |

**Global OOS +5,2% medio, 66% de ventanas positivas; estable en 5 semillas**
(media global positiva en las 5). FinBERT pesa 0,104 (10,4%) en el modelo.

**Lectura honesta.** El bot ML+LLM es **positivo de verdad fuera de muestra** con
noticias reales point-in-time, y competitivo con el QQQ en este periodo porque
esquiva parte del *bear* de 2022. Pero el grueso del retorno viene de estar long
QQQ ~98% del tiempo (prima de riesgo del índice) y la ventaja sobre el índice es
**dependiente del régimen** (en un bull puro restaría, como en el Tier 8.2). No es
alpha robusto; es gestión de riesgo con un aporte medible (~10%) del sentimiento.
Coherente con las secciones 7-9 y la hipótesis del mercado eficiente.

**Limitación.** El dataset gratuito cubre 2019-2023, por lo que el OOS es un split
interno 2021-2023 (un único *bear*) y no el lockbox de precios 2024-2025. Con más
historia de noticias (varios ciclos) podrían sacarse conclusiones de robustez más
fuertes.

## 11. Tier 11 — HRP y arbitraje estadístico de pares (PROBADAS y DESCARTADAS)

Dos sugerencias externas, implementadas y medidas con el mismo rigor (A/B pareado,
bootstrap CI, semilla fija, sin look-ahead) y **descartadas por no mejorar los
resultados**. El código de ambas se **revirtió** tras medir (no se integró en el
sistema); aquí queda la constancia del experimento.

- **Hierarchical Risk Parity (HRP).** **No es aplicable** a la mejor config del
  proyecto (`faber_qqq_park`): esa estrategia es *long/flat* sobre un único índice
  (QQQ o efectivo), nunca mantiene una cartera de ≥2 posiciones, así que HRP no
  tiene nada que repartir. Donde sí aplica (estrategia multi-activo `aggr_plus`),
  HRP repite el patrón del "sizing por riesgo" de la §5: recorta el peor drawdown
  (−5 a −14pp) pero baja el retorno en proporción parecida, sin mejorar el Sharpe.
  Además la base `aggr_plus` ya es muy inferior a `faber_qqq_park` (peor DD −58/
  −69% vs −22%; en lockbox pierde a todos los plazos y bate al 60/40 solo el 0-3%
  frente al 73-100% de Faber). La versión *literal* del consejo (HRP sobre las
  co-entradas de la misma barra) es además un no-op: el filtro de señal escalona
  las entradas y casi nunca hay ≥2 candidatos simultáneos. **Conclusión: ninguna
  versión de HRP mejora el mejor resultado.**
- **Arbitraje estadístico de pares (spread mean-reversion, long/short).** Sobre
  pares correlacionados (KO/PEP, GLD/SLV…), la estrategia market-neutral **pierde
  dinero** (−1.7% medio, Sharpe −0.30) y es **significativamente peor** que la
  estrategia direccional existente sobre los mismos activos (Δ_Sharpe −0.69,
  Δretorno −7.4pp, ambos con IC95% < 0). Reutilizar los modelos ML (XGB/LGBM/RF)
  sobre el target de reversión **no aporta edge**: con ML ≈ sin ML (z-score puro),
  diferencia no significativa. Su único rasgo positivo es el drawdown bajo propio
  de la neutralidad de mercado, pero perder despacio no es mejorar.

Coherente con §5, §7, el Tier 8 y §10: ni el control de riesgo por correlación (HRP),
ni el cambio de framing a pares, ni la reutilización de los modelos ML mejoran la
rentabilidad ajustada a riesgo sobre el baseline. **Ambas se descartan; el código
se revirtió y el sistema queda como estaba.**
