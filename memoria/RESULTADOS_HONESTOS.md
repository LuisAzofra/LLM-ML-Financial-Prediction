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
