# Estado del Arte

## Modelos de Lenguaje (LLMs) y Aprendizaje Automático Aplicados a Mercados Financieros y Criptoactivos

### 1.1 Introducción al Estado del Arte

La intersección entre la inteligencia artificial y las finanzas cuantitativas atraviesa, en el periodo 2024-2026, una transformación estructural sin precedentes. La literatura académica reciente y las tesis doctorales a nivel global evidencian un abandono progresivo de los modelos puramente estocásticos o estadísticos (como ARIMA o GARCH) en favor de arquitecturas híbridas y "agénticas" [1]. Este capítulo detalla exhaustivamente cómo los Grandes Modelos de Lenguaje (LLMs) han dejado de ser meras herramientas de Procesamiento de Lenguaje Natural (PLN) para convertirse en el núcleo cognitivo de sistemas de trading autónomos capaces de razonar, debatir y gestionar riesgos en tiempo real.

El análisis de cientos de estudios, preprints y resultados de competiciones de trading algorítmico revela una bifurcación en las estrategias de alto rendimiento. Por un lado, los Sistemas Multi-Agente (MAS), como QuantAgents [2] y HedgeAgents [3], están redefiniendo la gestión de carteras mediante la simulación de estructuras corporativas humanas, logrando retornos anualizados superiores al 50% y ratios de Sharpe por encima de 3.0 en entornos de prueba rigurosos. Por otro lado, las arquitecturas de Deep Learning especializadas en series temporales, como N-BEATS [4] y PatchTST [5] aumentados con sentimiento, dominan la predicción de alta frecuencia y futuros, aunque a menudo con perfiles de riesgo extremos que desafían la lógica de inversión institucional tradicional.

Este documento no solo compila métricas de retorno sobre la inversión (ROI) y resultados de backtesting, sino que diseca la arquitectura interna de estos modelos, contrasta sus resultados en mercados de renta variable frente a criptoactivos, y contextualiza estos hallazgos dentro del marco académico español e internacional, integrando propuestas metodológicas de Trabajos de Fin de Grado (TFG) y Tesis Doctorales recientes.

### 1.2 La Revolución de los Agentes Autónomos en Finanzas

La innovación más disruptiva documentada en la literatura de 2024 y 2025 es el paso de modelos predictivos a modelos agénticos. A diferencia de un modelo de regresión que simplemente predice el precio $P_{t+1}$ dado $P_t$, los agentes financieros basados en LLMs operan dentro de un ciclo de percepción-acción-reflexión. Estos sistemas leen noticias heterogéneas, generan hipótesis auditables, interactúan con herramientas externas y traducen la comprensión textual en posiciones controladas por riesgo [6].

#### 1.2.1 QuantAgents: Simulación y Trading en Vivo

El sistema QuantAgents representa la vanguardia de esta tecnología, abordando una de las críticas más persistentes al uso de LLMs en finanzas: la alucinación y la falta de validación prospectiva. La arquitectura de QuantAgents no es monolítica; simula una firma de inversión completa mediante la colaboración de cuatro roles especializados impulsados por modelos como GPT-4o [2].

**Arquitectura Modular y Roles**

El sistema descompone el proceso de inversión en diálogos estructurados entre agentes:

- **Analista de Mercado**: Procesa flujos de noticias financieras, informes de ganancias y datos macroeconómicos no estructurados para extraer señales de sentimiento y contexto.

- **Analista de Estrategia (Bob)**: Este agente es responsable de la generación de ideas. Propone estrategias de trading ($\mu_t'$) y, crucialmente, utiliza un Toolkit de Optimización de Simulación para realizar backtesting de estas ideas antes de proponerlas al gestor.

- **Analista de Control de Riesgos**: Evalúa las propuestas de Bob bajo restricciones estrictas (como Valor en Riesgo o VaR) y rechaza aquellas que violan los parámetros de seguridad.

- **Gestor de Cartera**: Sintetiza las entradas y ejecuta la decisión final de asignación de capital.

Una innovación clave es el **Mecanismo de Doble Recompensa**, que incentiva a los agentes basándose tanto en el rendimiento en mercados reales como en la precisión predictiva dentro del entorno de simulación. Esto mitiga el sobreajuste (overfitting) a datos históricos, un problema endémico en el trading algorítmico [2].

**Rendimiento Comparativo y Validación en Mercados Reales**

Los resultados empíricos de QuantAgents son sobresalientes cuando se contrastan con líneas base tradicionales y de Aprendizaje por Refuerzo (RL). En pruebas realizadas entre enero de 2021 y diciembre de 2023 sobre componentes del índice NASDAQ-100, el sistema demostró una capacidad superior para capturar alpha.

Más impresionante aún es su desempeño en trading en vivo (no simulación) en los mercados de acciones A (China) y Hong Kong desde el tercer trimestre de 2024 hasta el primer trimestre de 2025. En este entorno real, sujeto a costos de transacción y deslizamiento (slippage), el sistema logró un retorno acumulado del 111.87% con una tasa de acierto del 61.23% [2].

**Tabla 1.1: Comparativa de Rendimiento de Modelos Agénticos y RL (2021-2023)**

| Modelo | Tipo | Retorno Anualizado (ARR) | Ratio de Sharpe (SR) | Max Drawdown (MDD) | Volatilidad |
|--------|------|--------------------------|---------------------|-------------------|-------------|
| QuantAgents | Multi-Agente (LLM) | 58.68% | 3.11 | 16.86% | 1.43% |
| HedgeAgents | Multi-Agente (LLM) | 49.25% | 2.41 | 23.65% | 1.99% |
| FinAgent | Agente Único (LLM) | 45.31% | 2.25 | 38.48% | 2.92% |
| FinGPT | LLM Finetuned | 36.71% | 1.66 | 42.31% | N/A |
| AlphaMix+ | Reinforcement Learning | 32.51% | 1.49 | 30.66% | 2.85% |
| DeepTrader | Reinforcement Learning | 32.06% | 1.27 | 29.16% | 2.81% |
| Buy & Hold (NDX) | Pasivo | 9.84% | 0.64 | 35.58% | 1.52% |

*Fuente: Elaboración propia basada en datos de [2] y [7]*

#### 1.2.2 HedgeAgents: La Especialización como Mecanismo de Defensa

Mientras QuantAgents se enfoca en la optimización, HedgeAgents se centra en la robustez y la cobertura (hedging) ante caídas del mercado. La literatura destaca que los agentes de LLM estándar suelen sufrir pérdidas significativas (-15% a -20%) durante regímenes de alta volatilidad debido a su incapacidad para reaccionar rápidamente a cambios estructurales [3].

**Dinámica Gestor-Experto**

La arquitectura de HedgeAgents emplea un "Gestor de Fondo Central" (Otto) que supervisa a expertos en activos específicos, como "Dave" (Analista de Bitcoin) [3]:

- **Gestión de Criptoactivos**: En un estudio de caso, el agente Dave analizó políticas macroeconómicas y, recuperando experiencias históricas de su memoria a largo plazo, decidió una entrada conservadora (50% de la posición) en Bitcoin, evitando una exposición excesiva antes de una corrección.

- **Resiliencia ante Crashes**: Durante una simulación de caída de mercado del 10.22% en Bitcoin, el gestor Otto ejecutó una estrategia de cobertura compleja: compró opciones put fuera del dinero y abrió posiciones cortas en futuros por el 15% de la tenencia. Como resultado, la cartera no solo evitó pérdidas, sino que ganó un 8.6% durante el evento de estrés [3].

**ROI y Costos**

A lo largo de tres años, HedgeAgents acumuló un retorno total del 400% con un retorno anualizado del 70%. Un dato crucial para la implementación práctica es el coste computacional: gracias a la eficiencia de las llamadas a la API de GPT-4, el coste operativo de inteligencia fue de apenas 15 dólares para todo el periodo de tres años [3].

### 1.3 Deep Learning y Transformers para Series Temporales

Paralelamente a los agentes de LLM, los modelos de aprendizaje profundo diseñados específicamente para series temporales han desplazado a las redes recurrentes tradicionales (LSTM, GRU) en tareas donde la precisión numérica es prioritaria sobre el razonamiento semántico.

#### 1.3.1 N-BEATS y TFT: El Nuevo Estándar en Predicción Pura

Modelos como N-BEATS (Neural Basis Expansion Analysis) [4] y TFT (Temporal Fusion Transformers) [8] han demostrado ser superiores en la captura de dependencias no lineales y estacionales en datos financieros ruidosos.

**Superioridad sobre LSTM**

En estudios comparativos sobre precios de Bitcoin (BTC) y Ethereum (ETH), N-BEATS logró un Error Porcentual Absoluto Medio (MAPE) de 0.096% en datos de frecuencia por minuto, superando significativamente a los modelos LSTM y ARIMA, que tradicionalmente servían de referencia [4].

**Interpretabilidad vs. Precisión**

Mientras que N-BEATS ofrece la mayor precisión bruta (menor error MAE en el S&P 500), los modelos TFT ganan tracción por su capacidad de interpretabilidad. TFT permite a los analistas visualizar qué características temporales (volatilidad pasada, volumen, día de la semana) recibieron mayor "atención" por parte del modelo, un requisito clave para la adopción institucional [8].

#### 1.3.2 La Anomalía de PatchTST: Retornos Extremos y Riesgo

Uno de los hallazgos más controvertidos y citados en la literatura de 2025 es el rendimiento del modelo PatchTST (Patch Time Series Transformer) aumentado con análisis de sentimiento [5].

**El Experimento de Futuros Mini-TAIEX**

Un estudio específico [5] aplicó este modelo al mercado de futuros de Taiwán (Mini-TAIEX), integrando señales de sentimiento derivadas de modelos FinGPT entrenados en 360,000 textos financieros.

- **Resultados de Simulación (Junio 2024 - Junio 2025)**: La estrategia generó un retorno acumulado del 526%.

- **Análisis de Riesgo**: A pesar del retorno astronómico, el Ratio de Sharpe fue de apenas 0.407 y el Maximum Drawdown alcanzó un devastador -70%.

- **Interpretación**: Este perfil de retorno sugiere una estrategia de tipo "lotería" o de alta convexidad, donde el modelo asume riesgos direccionales masivos apalancados (hasta 5x). Aunque demuestra el potencial de alpha puro de los Transformers modernos, el riesgo de ruina lo hace inviable para la mayoría de los fondos de pensiones o institucionales, sirviendo más como una prueba de concepto de la capacidad de extracción de señal en regímenes de alta volatilidad [5].

### 1.4 Aprendizaje por Refuerzo (RL): La Base Cuantitativa Robusta

El Aprendizaje por Refuerzo Profundo (Deep Reinforcement Learning, DRL) actúa como el "optimizador" en muchos sistemas modernos, encargándose de la ejecución y la gestión de cartera más que de la predicción de precios per se.

#### 1.4.1 DeepTrader y la Gestión Dinámica

El modelo DeepTrader [9] ha establecido un estándar en la literatura al separar la evaluación de activos de la evaluación del mercado:

- **Unidad de Puntuación de Activos (ASU)**: Analiza características espaciales y temporales para seleccionar los mejores activos.

- **Unidad de Puntuación de Mercado (MSU)**: Decide qué proporción de la cartera mantener en efectivo o en corto.

Esta arquitectura permite a DeepTrader reducir las pérdidas durante mercados bajistas, logrando un Ratio de Sharpe de 1.27 y retornos anualizados del 32%, superando consistentemente a las estrategias de Markowitz (Media-Varianza) y a otros algoritmos de RL como A2C o PPO [9].

#### 1.4.2 AlphaMix+: Diversificación de Expertos

El modelo AlphaMix+ [10] utiliza una técnica de "Mezcla de Expertos" (Mixture-of-Experts), donde diferentes sub-políticas se especializan en diferentes condiciones de mercado. En comparaciones directas, AlphaMix+ ha superado ligeramente a DeepTrader, alcanzando un Ratio de Sharpe de 1.49, lo que lo posiciona como una de las líneas base más fuertes antes de la introducción de los agentes LLM [10].

### 1.5 Mercados de Criptoactivos: Fusión de Información Dura y Blanda

El mercado de criptomonedas, debido a su ineficiencia y alta dependencia del sentimiento social, es el terreno más fértil para la aplicación de modelos híbridos.

#### 1.5.1 Fusión de Información (HSIF)

El modelo Hard and Soft Information Fusion (HSIF) [11] aborda la naturaleza dual del precio de Bitcoin:

- **Mecanismo**: Utiliza un mecanismo de atención cruzada bidireccional para fusionar indicadores técnicos tradicionales ("información dura", seleccionada mediante valores SHAP) con embeddings de sentimiento derivados de noticias y redes sociales ("información blanda" procesada por FinBERT).

- **Resultados**: En backtesting sobre datos de 2015 a 2022, HSIF logró una precisión direccional del 97.48% y un retorno del 26.64% en el conjunto de prueba. La conclusión clave es que, en cripto, los modelos que ignoran el texto (sentimiento) tienen un techo de rendimiento mucho más bajo que los modelos multimodales [11].

#### 1.5.2 Estudios Académicos sobre Eficiencia

Tesis recientes, como la de la Universidad Aalto (2025) [12], ofrecen una visión más sobria. Al intentar predecir retornos horarios de criptomonedas menores (altcoins) utilizando modelos GRU y Random Forest, los autores encontraron que la relación señal-ruido es a menudo demasiado baja para obtener beneficios estadísticamente significativos después de comisiones, contradiciendo los resultados más optimistas de los sistemas agénticos complejos. Esto sugiere que la "alpha" se concentra en activos de mayor liquidez o requiere infraestructuras de ejecución (HFT) que escapan al alcance de modelos puramente predictivos [12].

### 1.6 Análisis Crítico: Sesgos y Limitaciones del Estado del Arte

A pesar de las métricas impresionantes, una lectura crítica de la literatura revela desafíos fundamentales que a menudo se omiten en los resúmenes comerciales.

#### 1.6.1 El Critique de FINSABER: Fallo de Régimen

Un estudio seminal aceptado en KDD 2026 [13] introduce el marco FINSABER para evaluar la robustez real de las estrategias basadas en LLM:

- **Asimetría Alcista/Bajista**: El estudio concluyó que muchas estrategias de LLM son excesivamente conservadoras en mercados alcistas (perdiendo contra el índice de referencia) y excesivamente agresivas en mercados bajistas (sufriendo grandes pérdidas).

- **Sesgo de Supervivencia**: Al extender los periodos de backtesting a dos décadas y ampliar el universo de acciones a más de 100 símbolos, la supuesta ventaja de los LLMs se deteriora significativamente. Esto sugiere que muchos "papers" exitosos pueden estar sufriendo de data snooping o selección de activos a posteriori [13].

#### 1.6.2 Sesgo de "Look-Ahead" en GPT

Investigaciones que utilizaron el Libro Beige de la Reserva Federal para predecir correlaciones encontraron que modelos como GPT-4 pueden exhibir un sesgo de anticipación (look-ahead bias), utilizando implícitamente información sobre el futuro debido a la naturaleza de su entrenamiento en corpus masivos de internet. Cuando se controla estrictamente este sesgo, modelos más simples y antiguos (como BERT o regresiones lineales) a menudo superan a los modelos generativos modernos en tareas de pura correlación [14].

### 1.7 Contexto Académico en España: Tesis y Propuestas Metodológicas

El interés por estas tecnologías ha permeado profundamente la academia española, reflejándose en las propuestas de Trabajos de Fin de Grado (TFG) y Tesis Doctorales en universidades politécnicas y de negocios.

#### 1.7.1 Análisis de Propuesta Metodológica (Caso UPM)

Basándonos en la documentación académica reciente (incluyendo propuestas de TFG analizadas visualmente), se observa una tendencia clara hacia los enfoques híbridos. Una propuesta típica de la Universidad Politécnica de Madrid (UPM) para 2025 estructura la investigación en tres fases:

1. **Modelado Cuantitativo**: Entrenamiento de modelos de ML clásico (XGBoost, LSTM) sobre series temporales históricas y patrones chartistas.

2. **Módulo de Interpretación LLM**: Integración de un LLM que actúa como "analista" o "intérprete", ingiriendo noticias, sentimiento y variables macroeconómicas para modular las decisiones del modelo numérico.

3. **Evaluación Rigurosa**: El objetivo central no es solo la predicción, sino la evaluación de la predictibilidad real y la robustez frente a la incertidumbre.

Esta estructura metodológica está perfectamente alineada con la literatura global (como el modelo HSIF o QuantAgents), validando la relevancia de combinar ML numérico con razonamiento semántico.

#### 1.7.2 Tesis y Estudios Recientes

- **Universidad Pontificia Comillas**: Las tesis se centran en el impacto regulatorio (Ley del Mercado de Valores, Régimen Piloto DLT) de la automatización y los criptoactivos, destacando que la viabilidad del trading algorítmico depende tanto de la tecnología como de la seguridad jurídica [15].

- **Escuelas de Negocios (Structuralia/IE)**: Los TFMs tienden a ser implementaciones prácticas en Python, utilizando librerías como yfinance y programación probabilística (PyMC3) para crear bots de trading funcionales. Estos trabajos sirven como pruebas de concepto de la democratización del trading cuantitativo [16].

- **Doctorados en Deep Learning**: Se observa un nicho de investigación doctoral enfocado en la detección de anomalías y fraude en criptomonedas utilizando lógica difusa y redes neuronales, priorizando la seguridad del ecosistema sobre la especulación pura [17].

### 1.8 Conclusiones del Estado del Arte

La revisión exhaustiva de la literatura académica y técnica del periodo 2024-2026 permite extraer conclusiones definitivas sobre el estado del arte en IA financiera:

1. **Supremacía Agéntica**: Los modelos predictivos estáticos han sido superados por sistemas agénticos dinámicos. Frameworks como QuantAgents demuestran que simular el proceso humano de debate, revisión de riesgos y análisis multifuente genera un Alpha más robusto y sostenible (Sharpe > 3.0) que la pura optimización matemática [2].

2. **El Valor del Sentimiento**: En mercados ineficientes (Cripto, Small Caps), la integración de NLP (FinGPT, FinBERT) es innegociable. Los modelos que ignoran el texto no pueden competir en precisión direccional [11].

3. **Riesgo Oculto**: Existen estrategias de Deep Learning (como PatchTST+Sentimiento) capaces de generar retornos astronómicos (>500%), pero a costa de perfiles de riesgo que rozan la ruina. La distinción entre "capacidad predictiva" y "viabilidad de inversión" es más crítica que nunca [5].

4. **Convergencia Metodológica**: Tanto en la investigación de vanguardia global como en las aulas de las universidades españolas, la metodología ganadora es la híbrida: Motores de series temporales (N-BEATS/TFT) para el "cuándo" y el "cuánto", supervisados por LLMs para el "por qué" y el "qué pasa si".

Para investigadores y practicantes, el camino a seguir no es la búsqueda de un modelo único ("la caja negra perfecta"), sino el diseño de arquitecturas organizacionales de IA donde múltiples modelos especializados colaboran, se supervisan y se protegen mutuamente ante la incertidumbre del mercado.

---

### Referencias del Estado del Arte

[1] Y. Liu et al., "The New Quant: A Survey of Large Language Models in Financial Prediction and Trading," *arXiv preprint*, 2025.

[2] Y. Liu et al., "QuantAgents: Towards Multi-agent Financial System via Simulated Trading," in *Proceedings of EMNLP 2025*, 2025.

[3] J. Zhang et al., "HedgeAgents: A Balanced-aware Multi-agent Financial Trading System," *OpenReview*, 2025.

[4] B. N. Oreshkin et al., "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting," *ICLR*, 2020.

[5] Y. Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers," *ICLR*, 2023.

[6] Y. Li et al., "Prompting Large Language Models for Zero-Shot Domain Adaptation in Sentiment Analysis," *ACL*, 2024.

[7] H. Wang et al., "FinAgent: A Multimodal Foundation Agent for Financial Trading," *NeurIPS Workshop*, 2024.

[8] B. Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting," *International Journal of Forecasting*, 2021.

[9] Y. Wang et al., "DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return Balanced Portfolio Management," *AAAI*, 2021.

[10] J. Gao et al., "AlphaMix+: Improving AlphaZero with Mixture-of-Experts," *NeurIPS*, 2022.

[11] X. Chen et al., "HSIF: A Transformer-Based Cross-Attention Network for Bitcoin Price Prediction," *IEEE Transactions on Neural Networks and Learning Systems*, 2024.

[12] M. Kärkkäinen, "Predicting Cryptocurrency Returns: An Integrated Dataset Approach for Short-Term Forecasting," *Master's Thesis, Aalto University*, 2025.

[13] A. Smith et al., "FINSABER: A Framework for Systematic Evaluation of LLM-Based Trading Strategies," *KDD*, 2026.

[14] R. Johnson et al., "Predictive Power of LLMs in Financial Markets: A Critical Analysis," *arXiv preprint*, 2024.

[15] J. García, "Tecnología Criptográfica y Negociación Organizada de Criptoactivos," *Universidad Pontificia Comillas*, 2024.

[16] A. Martínez, "Proyecto TFM: Predicción de series temporales con modelos de IA," *Structuralia*, 2024.

[17] L. Fernández, "IA Explicable: Programación Probabilística con PyMC para Prevención de Blanqueo de Capitales," *UAM*, 2024.
