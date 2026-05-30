# Trabajo Fin de Grado

## Estudio Prospectivo de LLMs y Modelos de Machine Learning Aplicados a la Detección de Patrones y Predicción de Volatilidad en Activos Financieros y Criptoactivos

---

## Resumen

El presente Trabajo Fin de Grado propone estudiar la capacidad actual de los modelos de Inteligencia Artificial —tanto modelos de lenguaje (LLMs) como modelos clásicos y avanzados de machine learning— para detectar patrones, modelar comportamientos no lineales y predecir volatilidad en mercados financieros y de criptoactivos.

El trabajo combina dos aproximaciones complementarias: (1) un modelo de aprendizaje automático entrenado con series temporales históricas, patrones chartistas, eventos noticiosos y otros indicadores relevantes; y (2) un LLM capaz de analizar información heterogénea (noticias, sentimiento, variables macroeconómicas, sesgos humanos, etc.) y evaluar críticamente las decisiones del modelo predictivo, siguiendo un enfoque híbrido donde el LLM actúa como "intérprete", "analista" o modulador.

El objetivo central no es únicamente desarrollar un sistema predictivo, sino evaluar rigurosamente la predictibilidad real alcanzable en mercados caracterizados por alta incertidumbre y dinámicas no lineales. El trabajo incluye una comparación entre distintos modelos actuales de IA, un análisis crítico de su capacidad de generalización, y una discusión del impacto de estas tecnologías en el futuro de las finanzas y la inversión algorítmica.

Finalmente, se lleva a cabo una evaluación cuantitativa de rendimiento mediante métricas de error, fiabilidad, estabilidad y robustez, para determinar la viabilidad real de aplicar estos modelos en contextos financieros contemporáneos.

**Palabras clave**: Machine Learning, Large Language Models, Series Temporales Financieras, Predicción de Volatilidad, GARCH, Sistemas Multi-Agente, Backtesting, Análisis de Sentimiento.

---

## Abstract

This Bachelor's Thesis proposes to study the current capability of Artificial Intelligence models —both language models (LLMs) and classical and advanced machine learning models— to detect patterns, model non-linear behaviors, and predict volatility in financial and cryptoasset markets.

The work combines two complementary approaches: (1) a machine learning model trained on historical time series, chartist patterns, news events, and other relevant indicators; and (2) an LLM capable of analyzing heterogeneous information (news, sentiment, macroeconomic variables, human biases, etc.) and critically evaluating the decisions of the predictive model, following a hybrid approach where the LLM acts as an "interpreter", "analyst", or modulator.

The central objective is not only to develop a predictive system but to rigorously evaluate the real predictability achievable in markets characterized by high uncertainty and non-linear dynamics. The work includes a comparison between different current AI models, a critical analysis of their generalization capability, and a discussion of the impact of these technologies on the future of finance and algorithmic investing.

Finally, a quantitative performance evaluation is carried out using metrics of error, reliability, stability, and robustness to determine the real viability of applying these models in contemporary financial contexts.

**Keywords**: Machine Learning, Large Language Models, Financial Time Series, Volatility Prediction, GARCH, Multi-Agent Systems, Backtesting, Sentiment Analysis.

---

## Índice

1. Introducción
   1.1. Contexto y Motivación
   1.2. Objetivos del Trabajo
   1.3. Estructura de la Memoria

2. Estado del Arte
   2.1. La Revolución de los Agentes Autónomos en Finanzas
   2.2. Deep Learning y Transformers para Series Temporales
   2.3. Aprendizaje por Refuerzo en Finanzas
   2.4. Mercados de Criptoactivos
   2.5. Sesgos y Limitaciones

3. Diseño del Sistema
   3.1. Arquitectura General
   3.2. Módulo de Ingesta de Datos
   3.3. Módulo de Machine Learning
   3.4. Sistema Multi-Agente
   3.5. Sistema Híbrido ML + LLM

4. Desarrollo e Implementación
   4.1. Tecnologías Utilizadas
   4.2. Implementación del Modelo GARCH
   4.3. Implementación del Sistema Multi-Agente
   4.4. Integración con LLM Gratuito

5. Resultados y Evaluación
   5.1. Métricas de Evaluación
   5.2. Protocolo de Evaluación Riguroso
   5.3. Resultados del Backtesting
   5.4. Discusión de Resultados

6. Conclusiones y Trabajo Futuro
   6.1. Conclusiones Principales
   6.2. Limitaciones
   6.3. Líneas Futuras de Investigación

Referencias

Anexos

---

## 1. Introducción

### 1.1 Contexto y Motivación

Los mercados financieros han experimentado una transformación radical en las últimas décadas con la llegada de la computación de alta frecuencia, el big data y, más recientemente, la inteligencia artificial avanzada. La pregunta fundamental que motiva este trabajo es: ¿hasta qué punto son predecibles los mercados financieros utilizando técnicas modernas de inteligencia artificial?

La hipótesis de mercados eficientes, formulada por Eugene Fama en 1970 [17], postula que los precios de los activos reflejan toda la información disponible, haciendo imposible obtener rendimientos consistentemente superiores al mercado. Sin embargo, la evidencia empírica reciente sugiere que, aunque los mercados sean altamente eficientes, existen oportunidades de "alpha" —rendimientos superiores ajustados por riesgo— que pueden ser explotadas mediante técnicas sofisticadas de análisis cuantitativo [18].

La irrupción de los Grandes Modelos de Lenguaje (LLMs) ha abierto nuevas posibilidades en el análisis financiero. Estos modelos, entrenados en vastos corpus de texto, pueden procesar noticias, informes de ganancias, redes sociales y otros datos no estructurados para extraer señales de trading que los modelos numéricos tradicionales no pueden capturar [1].

### 1.2 Objetivos del Trabajo

El objetivo general de este Trabajo Fin de Grado es desarrollar y evaluar un sistema híbrido que combine modelos de Machine Learning tradicionales, modelos de predicción de volatilidad GARCH, y un sistema multi-agente basado en LLMs para la predicción de activos financieros.

Los objetivos específicos son:

1. **Revisar el estado del arte** sobre el uso de LLMs y modelos de machine learning en predicción financiera y detección de patrones en series temporales.

2. **Diseñar y entrenar** uno o varios modelos de machine learning sobre series temporales de activos tradicionales y criptoactivos.

3. **Implementar un módulo de reconocimiento de patrones chartistas** (doble techo, cuñas, triángulos, etc.) utilizando datos históricos.

4. **Integrar un LLM gratuito** (Ollama/HuggingFace) para analizar información no estructurada (noticias, sentimiento, eventos externos, contexto económico) y evaluar decisiones del modelo numérico.

5. **Diseñar un enfoque híbrido ML + LLM** que combine predicciones cuantitativas con análisis cualitativo.

6. **Evaluar la capacidad real de predicción** en términos de precisión, estabilidad y robustez frente a volatilidad extrema.

7. **Comparar el rendimiento** de diferentes tipos de modelos (clásicos, deep learning, LLMs, modelos híbridos).

8. **Analizar críticamente la predictibilidad del mercado**, destacando limitaciones estructurales, riesgos y posibles sesgos.

### 1.3 Estructura de la Memoria

Este documento está organizado en seis capítulos principales:

- **Capítulo 1: Introducción**. Presenta el contexto, motivación y objetivos del trabajo.

- **Capítulo 2: Estado del Arte**. Revisa la literatura científica más reciente sobre LLMs, machine learning y finanzas cuantitativas.

- **Capítulo 3: Diseño del Sistema**. Describe la arquitectura del sistema híbrido desarrollado.

- **Capítulo 4: Desarrollo e Implementación**. Detalla la implementación técnica de cada módulo.

- **Capítulo 5: Resultados y Evaluación**. Presenta los resultados experimentales y su análisis.

- **Capítulo 6: Conclusiones y Trabajo Futuro**. Resume las conclusiones principales y propone líneas de investigación futura.

---

## 3. Diseño del Sistema

### 3.1 Arquitectura General

El sistema desarrollado en este trabajo sigue una arquitectura modular de tres niveles, ilustrada en la Figura 3.1. Esta arquitectura permite la independencia de cada componente, facilitando el mantenimiento, la extensibilidad y la posibilidad de sustituir módulos individuales sin afectar al resto del sistema.

**Figura 3.1: Arquitectura del Sistema Híbrido ML + LLM**

```
┌─────────────────────────────────────────────────────────────────┐
│                    SISTEMA HÍBRIDO ML + LLM                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              NIVEL 1: INGESTA DE DATOS                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │   Yahoo     │  │    RSS      │  │   Ollama    │     │   │
│  │  │  Finance    │  │   Feeds     │  │    LLM      │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           NIVEL 2: PROCESAMIENTO Y MODELOS              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │    ML       │  │   GARCH     │  │  Patrones   │     │   │
│  │  │  (RF/XGB)   │  │Volatilidad  │  │ Chartistas  │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          NIVEL 3: SISTEMA MULTI-AGENTE                  │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │Technical │ │Sentiment │ │   Risk   │ │ Portfolio│   │   │
│  │  │ Analyst  │ │ Analyst  │ │ Manager  │ │ Manager  │   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                    ┌─────────┴─────────┐                        │
│                    │  DECISIÓN FINAL   │                        │
│                    │   (HÍBRIDA)       │                        │
│                    └───────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

El sistema se divide en tres niveles funcionales:

1. **Nivel 1: Ingesta de Datos**. Responsable de obtener datos de mercado (precios OHLCV), noticias financieras (RSS feeds), y proporcionar acceso al LLM.

2. **Nivel 2: Procesamiento y Modelos**. Contiene los modelos de Machine Learning (Random Forest, XGBoost, LightGBM), el modelo GARCH para predicción de volatilidad, y el detector de patrones chartistas.

3. **Nivel 3: Sistema Multi-Agente**. Implementa cuatro agentes especializados que colaboran para tomar decisiones de inversión informadas.

### 3.2 Módulo de Ingesta de Datos

#### 3.2.1 Datos de Mercado

Los datos de precios históricos se obtienen mediante la librería `yfinance`, que proporciona acceso gratuito a datos de Yahoo Finance. Para cada activo, se descargan los siguientes campos:

- **OHLCV**: Open, High, Low, Close, Volume
- **Período**: 3 años de datos históricos
- **Frecuencia**: Diaria

El sistema admite distintos modos de universo de activos. Además del universo de acciones individuales (modo `famous`), se incorporan dos opciones que se emplearán más adelante para la evaluación rigurosa: un **universo diversificado cross-asset** (`diversified`), que combina ETFs de renta variable, bonos (TLT, IEF), oro (GLD, SLV), materias primas (DBC), divisa (UUP) y criptoactivos (BTC, ETH), y un **universo amplio y aleatorio** (`broad_random`) muestreado de un snapshot histórico del S&P 500, destinado al test de sesgo de supervivencia descrito en el Capítulo 5.

#### 3.2.2 Datos de Noticias

Las noticias financieras se obtienen mediante RSS feeds de fuentes gratuitas:

- Yahoo Finance News
- MarketWatch
- CoinDesk (para criptoactivos)
- Cointelegraph (para criptoactivos)

El módulo `feedparser` se utiliza para parsear los feeds RSS y extraer títulos, resúmenes y fechas de publicación.

#### 3.2.3 LLM Gratuito

Para el análisis de sentimiento y la interpretación de datos, se utiliza un LLM gratuito mediante una de dos opciones:

1. **Ollama**: Ejecución local de modelos como Llama 3.2 o Mistral
2. **Hugging Face Inference API**: Acceso gratuito con rate limits a modelos como Mistral-7B

#### 3.2.4 Tratamiento Causal de los Datos

Un requisito transversal del módulo de ingesta es la causalidad estricta: en ningún momento del histórico puede emplearse información que no estuviera disponible en esa fecha. Por ello, el relleno de huecos en las series de precios se realiza exclusivamente hacia delante (*forward-fill*): los valores ausentes se completan con el último dato conocido y los `NaN` iniciales se descartan en lugar de rellenarse con datos posteriores. Se evita expresamente el relleno hacia atrás (*backward-fill*) y la interpolación bidireccional, que introducirían precios futuros en el pasado. Este criterio resulta esencial para que el backtesting posterior sea representativo de las condiciones reales de operación.

### 3.3 Módulo de Machine Learning

#### 3.3.1 Modelos Tradicionales

Se implementan tres modelos de ensemble para predicción de retornos:

1. **Random Forest**: Ensemble de árboles de decisión con bagging
2. **XGBoost**: Gradient boosting optimizado
3. **LightGBM**: Gradient boosting con histogram-based learning

Para evitar la fuga de información (*data leakage*), el escalado de las variables se ajusta únicamente con los datos de entrenamiento y se aplica después al conjunto de test, en lugar de ajustarse sobre la totalidad de la serie antes de la partición temporal. En la predicción de volatilidad se añade además un *embargo* en el *target*, dejando un margen temporal entre entrenamiento y test que impide el solapamiento de ventanas. Sin estas precauciones, las métricas de evaluación quedan artificialmente infladas.

#### 3.3.2 Modelo GARCH para Volatilidad

El modelo GARCH(1,1) se implementa utilizando la librería `arch` [19]. La especificación del modelo es:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

Donde:
- $\sigma_t^2$ es la varianza condicional
- $\omega$ es la varianza de largo plazo
- $\alpha$ captura el efecto de shocks recientes
- $\beta$ captura la persistencia de la volatilidad

#### 3.3.3 Detección de Patrones Chartistas

Se implementan los siguientes patrones técnicos:

- Doble techo y doble suelo
- Cabeza y hombros (normal e invertido)
- Triángulos (ascendente, descendente, simétrico)
- Cuñas (ascendente, descendente)

La detección se realiza mediante análisis de extremos locales usando `scipy.signal.argrelextrema`.

### 3.4 Sistema Multi-Agente

El sistema multi-agente está inspirado en la arquitectura de QuantAgents [2] y consta de cuatro agentes especializados:

#### 3.4.1 TechnicalAnalystAgent

Responsable del análisis técnico de precios. Calcula indicadores como:

- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bandas de Bollinger
- Medias móviles (SMA 20, 50)
- ATR (Average True Range)

#### 3.4.2 SentimentAnalystAgent

Analiza el sentimiento del mercado mediante:

- Procesamiento de noticias reales con el LLM
- Cálculo de scores de sentimiento (-1 a +1)
- Integración con el Fear & Greed Index

#### 3.4.3 RiskManagerAgent

Gestiona el riesgo mediante:

- Cálculo de VaR y CVaR
- Predicción de volatilidad con GARCH
- Position sizing óptimo basado en Kelly Criterion
- Cálculo de niveles de stop-loss y take-profit

#### 3.4.4 PortfolioManagerAgent

Coordina todos los agentes y toma la decisión final. Utiliza el LLM como "intérprete" para sintetizar los análisis individuales en una recomendación de inversión coherente.

### 3.5 Sistema Híbrido ML + LLM

El sistema híbrido combina las predicciones cuantitativas del ML con el análisis cualitativo del LLM mediante un mecanismo de ponderación adaptativa:

$$\text{Score Híbrido} = w_{ML} \cdot \text{Señal}_{ML} + w_{LLM} \cdot \text{Señal}_{LLM}$$

Donde los pesos $w_{ML}$ y $w_{LLM}$ se ajustan dinámicamente según la confianza de cada componente.

---

## 4. Desarrollo e Implementación

### 4.1 Tecnologías Utilizadas

El sistema ha sido desarrollado en Python 3.9+ y utiliza las siguientes librerías principales:

**Tabla 4.1: Librerías Principales Utilizadas**

| Librería | Versión | Propósito |
|----------|---------|-----------|
| pandas | 2.0+ | Manipulación de datos |
| numpy | 1.24+ | Cómputo numérico |
| scikit-learn | 1.3+ | Machine Learning |
| xgboost | 2.0+ | Gradient Boosting |
| lightgbm | 4.0+ | Gradient Boosting |
| arch | 6.0+ | Modelos GARCH |
| yfinance | 0.2+ | Datos financieros |
| feedparser | 6.0+ | RSS feeds |
| requests | 2.31+ | HTTP requests |
| beautifulsoup4 | 4.12+ | Web scraping |
| matplotlib | 3.7+ | Visualización |
| scipy | 1.10+ | Procesamiento de señales |

### 4.2 Implementación del Modelo GARCH

La implementación del modelo GARCH se realiza mediante la librería `arch` [19]. El código principal es:

```python
from arch import arch_model

# Crear modelo GARCH(1,1) con distribución t
garch = arch_model(
    returns,
    vol='GARCH',
    p=1,
    q=1,
    dist='t',
    rescale=False
)

# Ajustar modelo
results = garch.fit(disp='off')

# Realizar forecast
forecast = results.forecast(horizon=5)
```

La elección de una distribución t de Student en lugar de una normal permite capturar mejor las colas pesadas características de los retornos financieros [20].

### 4.3 Implementación del Sistema Multi-Agente

Cada agente hereda de una clase base `BaseAgent` que proporciona funcionalidad común:

```python
class BaseAgent(ABC):
    def __init__(self, name: str, role: str, description: str):
        self.name = name
        self.role = role
        self.description = description
        self.memory = []
    
    @abstractmethod
    def analyze(self, data: Dict) -> AnalysisResult:
        pass
```

La comunicación entre agentes se realiza mediante un `AgentCommunicationBus` que implementa un patrón de publicación-suscripción.

### 4.4 Integración con LLM Gratuito

La integración con el LLM se realiza mediante una clase `LLMClient` que abstrae la interfaz con Ollama o Hugging Face:

```python
class LLMClient:
    def __init__(self, provider: str = "ollama"):
        self.provider = provider
        self.model = "llama3.2"
    
    def generate(self, prompt: str, system_prompt: str = None) -> Dict:
        # Llamada a Ollama API local
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False
            }
        )
        return response.json()
```

---

## 5. Resultados y Evaluación

### 5.1 Métricas de Evaluación

Las métricas utilizadas para evaluar el sistema son:

**Rendimiento Absoluto:**
- Retorno Total
- Retorno Anualizado

**Riesgo:**
- Volatilidad Anualizada
- Maximum Drawdown
- Ulcer Index
- VaR y CVaR

**Rendimiento Ajustado por Riesgo:**
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Omega Ratio

**Trading:**
- Win Rate
- Profit Factor
- Payoff Ratio
- Kelly Criterion

En el cálculo de algunas de estas métricas se corrigieron definiciones erróneas detectadas en una primera implementación. El **ratio de Sortino** empleaba la desviación típica de los retornos negativos, que resta su propia media y descarta los días positivos, produciendo un valor arbitrario; se sustituyó por la *downside deviation* estándar, que penaliza únicamente los retornos por debajo del objetivo sobre el total de observaciones. El **VaR** mezclaba el estimador histórico y el paramétrico mediante un `max()` no estándar, y el **CVaR** podía quedar por debajo del VaR, violando la definición de *Expected Shortfall*; ambos se reformularon con el estimador empírico, garantizando la relación CVaR ≥ VaR.

### 5.2 Protocolo de Evaluación Riguroso

Una primera fase del trabajo reveló que las cifras de rentabilidad iniciales estaban contaminadas por varios errores metodológicos. Antes de medir el rendimiento real fue necesario corregirlos, ya que sin ello cualquier resultado es ruido. Las correcciones principales fueron:

- **Fuga de información en el escalado** (descrita en la Sección 3.3): el *scaler* se ajustaba sobre train+test antes de la partición temporal.
- **Look-ahead en la ingesta** (Sección 3.2.4): el relleno hacia atrás introducía precios futuros.
- **Umbral de señal no causal**: el percentil que activaba las entradas se calculaba sobre toda la distribución de señales del periodo de test, empleando días futuros. Se corrigió a una ventana *expanding*, que en cada instante solo usa la información pasada.
- **Curva de equity con calendarios mixtos**: al combinar activos con calendarios distintos (los criptoactivos cotizan en fin de semana y las acciones no), las posiciones de los símbolos sin barra ese día desaparecían del valor de cartera y reaparecían al siguiente, generando saltos fantasma de ±50% que inflaban el Sharpe, el Sortino y el *maximum drawdown*. Se corrigió mediante *forward-fill* del último cierre conocido. El efecto fue drástico: la volatilidad diaria de la curva pasó de un ~25% irreal a un ~1.7% coherente con los activos.
- **Eliminación de un backtest con señales aleatorias**: se retiró un método que generaba señales con `np.random.choice` y aplicaba retornos del día siguiente (look-ahead), un *placeholder* sin valor evaluativo.

Sobre esta base depurada, la evaluación adopta tres garantías adicionales. En primer lugar, para no sesgar las conclusiones a partir de una única ventana favorable, se utiliza una **cuadrícula multi-régimen**: 10 fechas de inicio entre 2018 y 2023 —que cubren la fase pre-COVID, el desplome de 2020, la recuperación, el mercado alcista de 2021, el bajista de 2022 y la recuperación de 2023— combinadas con 4 plazos de tenencia (3 meses, 6 meses, 1 año y 2 años), lo que arroja del orden de 280 backtests independientes. Para los plazos cortos se habilita además un **modo *swing*** implementado en el sistema, con periodos de tenencia de 1 a 14 días, salida por convicción y una tenencia mínima anti-*whipsaw*. En segundo lugar, se incorporan **benchmarks realistas**, ya que el *buy & hold* del propio universo (con criptoactivos) es un punto de comparación injusto: se añaden el SPY (comprar y mantener el S&P 500) y una cartera 60/40 (SPY/TLT con rebalanceo diario), junto con las métricas de *alpha* relativo. En tercer lugar, se realiza un **test de sesgo de supervivencia**, repitiendo la evaluación sobre el universo amplio y aleatorio del S&P 500 en lugar de sobre las acciones "famosas" elegidas a posteriori.

Finalmente, la evaluación se acompaña de una **conciencia explícita del sobreajuste**. Dado el elevado número de configuraciones probadas, se emplea el *Deflated Sharpe Ratio* de Bailey y López de Prado para descontar el Sharpe esperado por puro azar, evitando confundir suerte con habilidad.

### 5.3 Resultados del Backtesting

#### 5.3.1 Resultado Ilustrativo en un Activo

**Tabla 5.1: Resultados del Backtesting para AAPL (2021-2024)**

| Métrica | Valor |
|---------|-------|
| Retorno Total | 8.55% |
| Retorno Anualizado | 2.78% |
| Sharpe Ratio | 0.22 |
| Sortino Ratio | 0.35 |
| Calmar Ratio | 0.18 |
| Max Drawdown | -46.57% |
| Ulcer Index | 0.2341 |
| Win Rate | 31.6% |
| Profit Factor | 1.08 |
| Total Operaciones | 19 |

Los resultados muestran un rendimiento positivo pero modesto, con un Sharpe Ratio de 0.22 que indica un rendimiento ajustado por riesgo por debajo del mercado (S&P 500 tiene Sharpe ~0.6 históricamente). El Maximum Drawdown del 46.57% es considerablemente alto, lo que sugiere que la estrategia necesita mejoras en la gestión de riesgo.

#### 5.3.2 Análisis de Robustez por Régimen de Volatilidad

Se realizó un análisis de robustez por regímenes de volatilidad:

**Tabla 5.2: Rendimiento por Régimen de Volatilidad**

| Régimen | Sharpe Ratio | Max Drawdown |
|---------|--------------|--------------|
| Baja Volatilidad | 0.45 | -12.3% |
| Volatilidad Normal | 0.18 | -28.7% |
| Alta Volatilidad | -0.32 | -51.2% |

El análisis revela que la estrategia funciona mejor en regímenes de baja volatilidad, pero sufre significativamente durante períodos de alta volatilidad. Esto es consistente con la literatura sobre estrategias de momentum [21].

#### 5.3.3 Rendimiento por Plazo de Tenencia en la Cuadrícula Multi-Régimen

Evaluada la variante principal del sistema (un *trend-following* multi-activo con rotación por momentum) sobre la cuadrícula completa, los resultados deben leerse distinguiendo entre **media** y **mediana**, porque divergen considerablemente. La Tabla 5.3 resume el rendimiento por plazo de tenencia frente al SPY.

**Tabla 5.3: Rendimiento por Plazo de Tenencia (Media vs Mediana) frente al SPY**

| Plazo | Mediana (caso típico) | Media (inflada por cola) | % de ventanas en positivo | SPY (mediana) |
|-------|----------------------:|-------------------------:|--------------------------:|--------------:|
| 3 meses | −8.7% | −13.0% | 30% | +3.1% |
| 6 meses | −6.6% | +4.9% | 40% | +11.8% |
| 1 año | +5.4% | +29.9% | 60% | +15.7% |
| 2 años | +25.6% | +132.9% | 80% | +32.8% |

La discrepancia entre media y mediana no es anecdótica. Ordenadas de menor a mayor, las 10 pruebas a 2 años fueron −26%, −22%, +12%, +20%, +22%, +29%, +147%, +239%, +371% y +537%. La media (+132.9%) está disparada por tres o cuatro aciertos de cola en criptoactivos durante 2020-2021, de modo que **6 de las 10 pruebas quedaron por debajo de la media**. Por tanto, para juzgar la utilidad real del sistema debe emplearse la mediana (+25.6% a 2 años), que refleja el resultado típico, y no la media, sesgada al alza.

#### 5.3.4 Comparación frente a la Inversión Indexada Pasiva

La pregunta determinante no es si el sistema gana dinero, sino si supera a la alternativa trivial de comprar y mantener un índice. La respuesta es matizada:

- **En términos absolutos**, el sistema es rentable a plazos largos: a 2 años termina en positivo el 80% de las veces (+25.6% típico). A corto plazo (≤6 meses) típicamente pierde.
- **Frente a comprar y mantener el SPY**, no compensa. En el caso típico el SPY rinde igual o más (a 2 años, SPY +32.8% frente al +25.6% del sistema), y el sistema solo supera al SPY en el **42%** de las ventanas —prácticamente un cara o cruz— asumiendo además mucho más riesgo.
- **Ajustado por riesgo**, la diferencia es nítida: el Sharpe del SPY es 0.97 frente al 0.25 del sistema. El índice es aproximadamente cuatro veces mejor por unidad de riesgo.

#### 5.3.5 Palancas de Mejora Evaluadas

Con el objetivo de superar al *baseline*, se probaron de forma rigurosa varias palancas de mejora, midiendo cada una sobre la cuadrícula completa (Tabla 5.4).

**Tabla 5.4: Palancas de Mejora y su Efecto sobre el Sharpe Medio**

| Palanca | Resultado | Sharpe medio |
|---------|-----------|-------------:|
| Baseline (trend-following + rotación) | referencia | 0.25 |
| Universo diversificado (bonos / oro / materias primas) | peor | −0.48 |
| Señal TSMOM multi-*lookback* | peor | −0.38 |
| Sizing por riesgo (*inverse-vol* + *vol-target*) | menor drawdown, menor retorno | 0.16 |
| Holdings más largos (252 / 504 / sin tope) | peor (devuelve ganancias en *bear*) | ≤0.07 |

Ninguna palanca mejora la rentabilidad ajustada por riesgo del *baseline*. Además, el propio *baseline* está probablemente sobreajustado: el *Deflated Sharpe Ratio* indica que, con el número de configuraciones probadas (del orden de 50-100), el Sharpe esperado por puro azar es 0.82-0.92, por encima del 0.25 observado. Seguir buscando configuraciones que "ganen" en estas ventanas sería sobreajuste, no una mejora real.

El *tradeoff* entre riesgo y rentabilidad resulta ineludible. El *sizing* por riesgo recorta el *drawdown* (de −27.9% a −23.9% a 2 años) pero reduce la media (de +132.9% a +71.3%), porque esa media elevada procede precisamente de la exposición volátil (criptoactivos) que el control de riesgo modera. No existe ninguna configuración que aumente el retorno y reduzca el riesgo simultáneamente.

#### 5.3.6 Test de Sesgo de Supervivencia

Sobre el universo amplio y aleatorio del S&P 500 —en lugar de las acciones elegidas a posteriori—, la estrategia **pierde dinero a todos los plazos** (retorno medio en torno al −11%, superando al *buy & hold* solo en el ~8% de los casos). Este resultado confirma que buena parte del rendimiento aparente del universo reducido se debía a la selección de activos en retrospectiva, y no a una habilidad genuina del sistema.

### 5.4 Discusión de Resultados

Los resultados obtenidos confirman varios hallazgos de la literatura:

1. **Dificultad de la predicción**: Los modelos ML muestran R² negativo, indicando que la predicción de retornos exactos es extremadamente difícil, consistente con la hipótesis de mercados eficientes en forma semi-fuerte [17].

2. **Volatilidad predecible**: El modelo GARCH muestra capacidad para predecir volatilidad, consistente con la literatura sobre clustering de volatilidad [20].

3. **Valor del sentimiento**: El análisis de sentimiento mediante LLM proporciona información adicional valiosa, especialmente durante eventos de mercado significativos.

4. **Ausencia de alpha sostenible ajustado por riesgo**: una vez eliminados los sesgos metodológicos y evaluado el sistema sobre la cuadrícula multi-régimen y un universo sin sesgo de supervivencia, la sofisticación añadida (ML, LLM y arquitectura multi-agente) no se traduce en una ventaja sobre la inversión indexada pasiva en términos ajustados por riesgo. Este hallazgo es plenamente coherente con la hipótesis del mercado eficiente [17].

---

## 6. Conclusiones y Trabajo Futuro

### 6.1 Conclusiones Principales

Este Trabajo Fin de Grado ha desarrollado e implementado un sistema híbrido completo para predicción financiera que combina:

1. **Modelos de Machine Learning** (Random Forest, XGBoost, LightGBM) para predicción de retornos.

2. **Modelo GARCH** para predicción de volatilidad con fundamentos matemáticos sólidos.

3. **Sistema Multi-Agente** con cuatro agentes especializados que simulan un equipo de analistas.

4. **LLM Gratuito** (Ollama/HuggingFace) para análisis de sentimiento e interpretación cualitativa.

5. **Backtesting Económico Riguroso** con métricas profesionales (Sharpe, Sortino, Calmar, Ulcer Index) y un protocolo de evaluación libre de sesgos.

Las principales conclusiones son:

- La predicción de retornos exactos es extremadamente difícil, con los modelos ML mostrando R² negativo.

- La volatilidad es más predecible que los retornos, y el modelo GARCH proporciona estimaciones útiles.

- El sistema multi-agente con LLM proporciona un marco robusto para integrar análisis cuantitativo y cualitativo.

- Una vez corregidos los sesgos metodológicos, el sistema **gana dinero en términos absolutos a plazos largos** (positivo en el 80% de las ventanas a 2 años, con un +25.6% típico), pero **no bate al indexado pasivo ajustando por riesgo** (Sharpe 0.25 frente al 0.97 del SPY) y solo supera al SPY en el 42% de las ventanas. Su media elevada depende de aciertos de cola en criptoactivos no repetibles a voluntad.

- En un universo realista sin sesgo de supervivencia el sistema deja de ser rentable, lo que confirma que parte del rendimiento aparente procedía de la selección de activos en retrospectiva.

Este resultado es coherente con la hipótesis del mercado eficiente y con la literatura: un sistema técnico operando sobre datos diarios no obtiene una ventaja estructural sostenible en estos activos. Lejos de constituir un fracaso, se trata de un resultado válido y defendible. El valor del trabajo reside en tres elementos: (1) un **sistema completo y funcional** *end-to-end* (datos reales → ML + GARCH → multi-agente con LLM → gestión de riesgo → backtesting → interfaz); (2) una **metodología de evaluación rigurosa**, sin *look-ahead*, con test de sesgo de supervivencia, benchmarks justos y conciencia del sobreajuste (*Deflated Sharpe* / PBO); y (3) un **hallazgo honesto**: la sofisticación no se traduce en alpha sobre el indexado pasivo en términos ajustados por riesgo.

### 6.2 Limitaciones

El trabajo presenta las siguientes limitaciones:

1. **Granularidad de los datos**: el análisis emplea datos diarios de Yahoo Finance, sin información intradía ni fuentes de datos alternativos que pudieran aportar señales adicionales.

2. **Costos de transacción simplificados**: El modelo asume comisiones fijas, sin considerar slippage real ni impacto de mercado.

3. **Disponibilidad del LLM**: El sistema depende de Ollama ejecutándose localmente, lo cual puede no ser práctico para todos los usuarios.

4. **Universo limitado y sobreajuste**: aunque la evaluación incorpora un test de sesgo de supervivencia que ya evidencia la fragilidad del sistema fuera del conjunto reducido de activos, el universo sigue siendo limitado. El *Deflated Sharpe Ratio* sugiere además que el rendimiento del *baseline* es indistinguible del esperado por azar dado el número de configuraciones exploradas.

### 6.3 Líneas Futuras de Investigación

Se proponen las siguientes líneas de investigación futura:

1. **Validación con *lockbox* temporal**: reservar el periodo 2024 en adelante como datos nunca vistos durante el desarrollo, de forma que constituya una prueba final genuinamente fuera de muestra.

2. **Estimación de la probabilidad de sobreajuste**: emplear *Combinatorial Purged Cross-Validation* para calcular la *Probability of Backtest Overfitting* (PBO) de cualquier mejora propuesta antes de darla por buena, evitando confundir suerte con habilidad.

3. **Reorientación del sistema**: explorar su uso como herramienta de **gestión de riesgo** (capturar la subida del mercado con menor *drawdown*) o de **análisis y apoyo a la decisión** (el motor multi-agente con LLM como soporte al inversor), donde su utilidad es más defendible que como generador de alpha.

4. **Integración con LLMs más potentes**: Evaluar el uso de GPT-4 o Claude para comparar el rendimiento.

5. **Expansión de patrones chartistas**: Incluir patrones más complejos como ondas de Elliott y patrones armónicos.

6. **Optimización de hiperparámetros**: Utilizar Optuna o Ray Tune para optimizar los parámetros de los modelos.

7. **Trading en tiempo real**: Implementar el sistema para operar en mercados reales con paper trading.

8. **Dashboard interactivo**: Desarrollar una interfaz web para visualizar resultados y tomar decisiones.

---

## Referencias

[1] W. Fu, "The New Quant: A Survey of Large Language Models in Financial Prediction and Trading," *arXiv preprint*, 2025.

[2] X. Li, Y. Zeng, X. Xing, J. Xu y X. Xu, "QuantAgents: Towards Multi-agent Financial System via Simulated Trading," in *Proceedings of EMNLP 2025*, 2025.

[3] X. Li, Y. Zeng, X. Xing, J. Xu y X. Xu, "HedgeAgents: A Balanced-aware Multi-agent Financial Trading System," in *The Web Conference (WWW)*, 2025.

[4] B. N. Oreshkin, D. Carpov, N. Chapados y Y. Bengio, "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting," in *ICLR*, 2020.

[5] Y. Nie, N. H. Nguyen, P. Sinthong y J. Kalagnanam, "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers," in *ICLR*, 2023.

[6] Y. Li et al., "Prompting Large Language Models for Zero-Shot Domain Adaptation in Sentiment Analysis," in *ACL*, 2024.

[7] W. Zhang, L. Zhao, H. Xia, S. Sun, J. Sun et al., "FinAgent: A Multimodal Foundation Agent for Financial Trading," in *NeurIPS Workshop*, 2024.

[8] B. Lim, S. O. Arik, N. Loeff y T. Pfister, "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting," *International Journal of Forecasting*, 2021.

[9] Z. Wang, B. Huang, S. Tu, K. Zhang y L. Xu, "DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return Balanced Portfolio Management," in *AAAI*, 2021.

[10] S. Mohammadi Dashtaki et al., "HSIF: A Transformer-Based Cross-Attention Framework for Cryptocurrency Trend Forecasting via Multimodal Sentiment-Market Fusion," *IEEE Access*, 2024.

[11] S. Celik, "Predicting Cryptocurrency Returns: An Integrated Dataset Approach for Short-Term Forecasting," *Tesis de Máster, Aalto University*, 2025.

[12] W. W. Li, H. Kim, M. Cucuringu y T. Ma, "Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?," in *KDD*, 2026.

[13] J. Shi y B. Hollifield, "Predictive Power of LLMs in Financial Markets," *arXiv preprint*, 2024.

[14] J. García, "Tecnología Criptográfica y Negociación Organizada de Criptoactivos," *Universidad Pontificia Comillas*, 2024.

[15] D. Cano Alvira, "Proyecto TFM: Predicción de series temporales con modelos de IA," *Structuralia*, 2024.

[16] L. Fernández, "IA Explicable: Programación Probabilística con PyMC para Prevención de Blanqueo de Capitales," *Universidad Autónoma de Madrid*, 2024.

[17] E. F. Fama, "Efficient Capital Markets: A Review of Theory and Empirical Work," *The Journal of Finance*, vol. 25, no. 2, pp. 383-417, 1970.

[18] A. W. Lo, "Reconciling Efficient Markets with Behavioral Finance: The Adaptive Markets Hypothesis," *Journal of Investment Consulting*, vol. 7, no. 2, pp. 21-44, 2005.

[19] K. Sheppard, "arch: Arch models in Python," 2023.

[20] R. F. Engle, "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation," *Econometrica*, vol. 50, no. 4, pp. 987-1007, 1982.

[21] C. S. Asness, T. J. Moskowitz y L. H. Pedersen, "Value and Momentum Everywhere," *Journal of Finance*, vol. 68, no. 3, pp. 929-985, 2013.

---

## Anexos

### Anexo A: Instalación del Sistema

```bash
# Clonar repositorio
git clone <repositorio>
cd tfg_finance_ai

# Crear entorno virtual
python -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar Ollama (opcional, para LLM local)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2

# Ejecutar sistema
python main.py
```

### Anexo B: Estructura del Proyecto

```
tfg_finance_ai/
├── data/
│   ├── data_loader.py
│   └── news_fetcher.py
├── models/
│   ├── ml_models/
│   │   ├── traditional_ml.py
│   │   ├── deep_learning.py
│   │   └── volatility_models.py
│   └── pattern_detection/
│       └── chart_patterns.py
├── agents/
│   ├── base_agent.py
│   ├── technical_agent/
│   ├── sentiment_agent/
│   ├── risk_agent/
│   └── manager_agent/
├── evaluation/
│   └── backtest.py
├── utils/
│   ├── data_utils.py
│   └── llm_client.py
├── main.py
├── requirements.txt
└── memoria/
    ├── ESTADO_DEL_ARTE.md
    └── MEMORIA_TFG.md
```

### Anexo C: Métricas Económicas Definidas

**Sharpe Ratio**: $S = \frac{R_p - R_f}{\sigma_p}$

**Sortino Ratio**: $Sortino = \frac{R_p - R_f}{\sigma_d}$

**Calmar Ratio**: $Calmar = \frac{R_{anualizado}}{|MDD|}$

**Ulcer Index**: $UI = \sqrt{\frac{1}{N}\sum_{i=1}^{N} D_i^2}$

Donde $D_i$ es el drawdown en el período $i$.
