# Estudio Prospectivo de LLMs y ML en Predicción Financiera

## Trabajo Fin de Grado - Versión 2.0

**Sistema Híbrido con Datos Reales, LLM Gratuito y GARCH**

Este proyecto implementa un sistema híbrido completo que combina modelos de Machine Learning tradicionales, modelos GARCH para predicción de volatilidad, y un sistema multi-agente basado en LLMs gratuitos (Ollama/HuggingFace) para la predicción de activos financieros y criptoactivos.

---

## 🎯 Características Principales

### ✅ Datos Reales (Sin Mocks)
- **Precios históricos**: Yahoo Finance API
- **Noticias financieras**: RSS feeds de fuentes reales (Yahoo Finance, MarketWatch, CoinDesk)
- **Indicadores de sentimiento**: Fear & Greed Index

### ✅ LLM / NLP Gratuito
- **FinBERT** (`ProsusAI/finbert`): modelo Transformer **finetuneado** sobre texto financiero — proveedor principal del agente de sentimiento (ver `utils/finbert_sentiment.py`)
- **Ollama**: Ejecución local de Llama 3.2, Mistral
- **Hugging Face**: API gratuita con rate limits
- **Fallback**: Análisis basado en reglas si no hay LLM disponible

### ✅ Predicción de Volatilidad Matemática
- **GARCH(1,1)**: Modelo con fundamentos estadísticos sólidos
- **EGARCH**: Captura efectos asimétricos
- **Forecast con intervalos de confianza**

### ✅ Sistema Multi-Agente
- **TechnicalAnalystAgent**: Indicadores técnicos
- **SentimentAnalystAgent**: Análisis de sentimiento con LLM real
- **RiskManagerAgent**: VaR, CVaR, GARCH, position sizing
- **PortfolioManagerAgent**: LLM como intérprete

### ✅ Backtesting Económico Riguroso
- **Métricas profesionales**: Sharpe, Sortino, Calmar, Ulcer Index
- **Análisis de robustez por regímenes**
- **Testing de sensibilidad a parámetros**

---

## 📁 Estructura del Proyecto

```
.
├── api.py                      # Servidor Flask: API REST + sirve el frontend
├── frontend/                   # Interfaz web (HTML/CSS/JS)
├── main.py                     # Análisis completo por consola (genera results/)
├── data/                       # Ingesta de datos
│   ├── data_loader.py          # Yahoo Finance
│   ├── news_fetcher.py         # RSS feeds reales
│   ├── news_sentiment_nasdaq.csv  # Sentimiento diario FinBERT (point-in-time)
│   └── universe_snapshots/     # Constituyentes S&P 500 (test de supervivencia)
├── models/
│   ├── ml_models/
│   │   ├── traditional_ml.py   # RF, XGBoost, LightGBM
│   │   ├── deep_learning.py    # LSTM, GRU
│   │   ├── volatility_models.py # GARCH, EGARCH
│   │   └── meta_labeling.py    # Meta-labeling (López de Prado)
│   └── pattern_detection/
│       └── chart_patterns.py   # Patrones chartistas
├── agents/
│   ├── base_agent.py
│   ├── debate.py               # Debate bull/bear + juez LLM
│   ├── technical_agent/        # Análisis técnico
│   ├── sentiment_agent/        # FinBERT/LLM + noticias reales
│   ├── risk_agent/             # GARCH + VaR
│   └── manager_agent/          # LLM intérprete
├── hybrid_system/              # Integración ML + LLM
├── trading_bot/                # Bot de paper trading + risk gate
├── evaluation/
│   └── backtest.py             # Métricas económicas y robustez
├── utils/                      # Cliente LLM, FinBERT, cachés, métricas, calibración
├── tools/                      # Experimentos, barridos y dashboards reproducibles
├── compare_*.py                # Comparativas A/B contra la API local
├── tests/                      # Tests de causalidad (sin look-ahead) y meta-labeling
├── notebooks/                  # Notebook de análisis
├── memoria/                    # Memoria, estado del arte y resultados honestos
├── PROGRESS.md                 # Registro de cada experimento (Tiers 1-11)
└── requirements.txt
```

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/LuisAzofra/LLM-ML-Financial-Prediction.git
cd LLM-ML-Financial-Prediction
```

### 2. Crear entorno virtual

Desarrollado y probado con Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Instalar Ollama (opcional, para LLM local)

```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Descargar desde https://ollama.com/download

# Descargar modelo (por defecto el sistema usa qwen2.5)
ollama pull qwen2.5
```

### 5. Verificar instalación

```bash
python -c "import arch; import yfinance; print('✅ Todo instalado correctamente')"
```

---

## 📊 Uso

### Interfaz web (API + frontend)

```bash
python api.py              # http://localhost:5000
PORT=5057 python api.py    # puerto usado por los scripts compare_*.py y tools/
```

### Análisis completo por consola

```bash
python main.py
```

### Usar el notebook de análisis

```bash
jupyter notebook notebooks/analisis_completo.ipynb
```

### Ejemplo de uso programático

```python
from data.data_loader import FinancialDataLoader
from data.news_fetcher import NewsFetcher
from models.ml_models.volatility_models import GARCHVolatilityModel
from utils.llm_client import LLMClient

# Cargar datos
loader = FinancialDataLoader()
df = loader.download_stock_data('AAPL', '2020-01-01', '2024-01-01')

# Obtener noticias reales
fetcher = NewsFetcher()
news = fetcher.fetch_yahoo_finance_news('AAPL', max_items=10)

# Predecir volatilidad con GARCH
returns = df['Close'].pct_change().dropna().values
garch = GARCHVolatilityModel(p=1, q=1)
garch.fit(returns)
forecast = garch.forecast(horizon=5)
print(f"Volatilidad predicha: {forecast.forecast_value*100:.2f}%")

# Analizar sentimiento con LLM
llm = LLMClient(provider="ollama")
result = llm.analyze_sentiment("Apple reports record earnings...")
print(f"Sentimiento: {result['parsed']['sentiment']}")
```

---

## 🔧 Configuración del LLM

El proveedor se elige con la variable de entorno `TFG_LLM_PROVIDER`:

| Proveedor | Modelo por defecto | Notas |
|-----------|--------------------|-------|
| `ollama` (default de la API) | `qwen2.5` | Requiere `ollama serve` en `localhost:11434` |
| `local` | `Qwen/Qwen2.5-0.5B-Instruct` | `transformers` en local; cambiable con `TFG_LOCAL_LLM` |
| `local-gguf` | Qwen2.5-1.5B-Instruct Q4 (GGUF) | Requiere `llama-cpp-python`; fichero en `TFG_LOCAL_GGUF_FILE` |
| `huggingface` | `mistralai/Mistral-7B-Instruct-v0.2` | API de inferencia de Hugging Face; cambiable con `TFG_HF_MODEL` |

```bash
ollama serve                                   # en una terminal
TFG_LLM_PROVIDER=ollama python api.py          # en otra
```

Otras variables útiles:

- `FINBERT_MODEL` (default `ProsusAI/finbert`): modelo de sentimiento financiero.
- `TFG_LIGHTWEIGHT=1` / `TFG_DISABLE_TF=1`: arranque ligero sin TensorFlow (sin LSTM/GRU).
- `PRICE_CACHE=1`: usa la caché local de precios (útil para tests reproducibles).

Si no hay LLM disponible, el sistema cae a análisis basado en reglas (VADER, palabras clave).

---

## 📈 Métricas Implementadas

### Rendimiento
- Retorno Total
- Retorno Anualizado

### Riesgo
- Volatilidad Anualizada
- Maximum Drawdown
- Ulcer Index
- VaR y CVaR

### Riesgo-Ajustado
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Omega Ratio

### Trading
- Win Rate
- Profit Factor
- Payoff Ratio
- Kelly Criterion

---

## 📚 Memoria del TFG

La memoria completa del TFG está disponible en:

- **[Estado del Arte](memoria/ESTADO_DEL_ARTE.md)**: Revisión exhaustiva de la literatura científica 2024-2026
- **[Memoria TFG](memoria/MEMORIA_TFG.md)**: Documento completo con diseño, implementación y resultados
- **[Resultados honestos](memoria/RESULTADOS_HONESTOS.md)**: Evaluación sin look-ahead, OOS y lockbox
- **[Bot ML+LLM con noticias](memoria/RESULTADOS_ML_LLM_NOTICIAS.md)**: FinBERT + RandomForest point-in-time
- **[PROGRESS](PROGRESS.md)**: Registro de cada experimento, aceptado o descartado

---

## 🧪 Testing

### Tests automáticos

Verifican que no hay look-ahead (invariancia al truncar el histórico) y el meta-labeling:

```bash
PYTHONPATH=. python tests/test_causality.py
PYTHONPATH=. python tests/test_meta_labeling.py
```

### Ejecutar tests de robustez

```python
from evaluation.backtest import Backtester, RobustnessTester

backtester = Backtester()
robustness_tester = RobustnessTester(backtester)

# Test por regímenes
regime_results = robustness_tester.test_regime_robustness(
    df, signal_generator, 'AAPL'
)

# Calcular score de robustez
score = robustness_tester.calculate_robustness_score(regime_results)
print(f"Score de robustez: {score['overall']:.2f}")
```

---

## 📊 Resultados Esperados

`main.py` produce los siguientes outputs:

```
results/
├── REPORTE_FINAL_<SÍMBOLO>.txt  # Reporte detallado por activo
├── backtest_analysis.png        # Gráficos de backtest
└── price_chart.png              # Gráficos de precios
```

Los ficheros incluidos en `results/` y `RESUMEN_EJECUCION.txt` son una ejecución de
ejemplo de la primera versión (febrero 2026). Los resultados actuales y rigurosos
están en `memoria/RESULTADOS_HONESTOS.md`.

---

## 📈 Resultados reales (evaluación honesta)

Evaluación rigurosa (sin look-ahead, fuera de muestra, *Deflated Sharpe*). Detalle
completo en [`memoria/RESULTADOS_HONESTOS.md`](memoria/RESULTADOS_HONESTOS.md),
[`memoria/RESULTADOS_ML_LLM_NOTICIAS.md`](memoria/RESULTADOS_ML_LLM_NOTICIAS.md) y
[`PROGRESS.md`](PROGRESS.md) (Tiers 7-10).

- **Hallazgo principal**: la sofisticación (ML + LLM + multi-agente) **no genera
  alpha** sobre el indexado pasivo ajustando por riesgo — coherente con la
  hipótesis del mercado eficiente. Resultado negativo válido y defendible.
- **Bot ML+LLM con noticias reales *point-in-time* (FinBERT + RandomForest)** →
  **retornos positivos *out-of-sample***. Entreno 2019-2020, test 2021-2023:

  | Plazo | ML+LLM (medio) | % ventanas positivas | QQQ comprar&mantener |
  |-------|---------------:|:--------------------:|---------------------:|
  | 1 mes  | +0,04% | 62% | −0,30% |
  | 6 meses| +7,33% | 62% | +5,50% |
  | 1 año  | +7,72% | 62% | +4,68% |
  | 2 años | +5,72% | 75% | +0,84% |

  Global OOS **+5,2% medio, 66% de ventanas positivas** (estable en 5 semillas).
  Honesto: el grueso del retorno viene de la exposición *long/flat* disciplinada al
  índice; FinBERT (LLM) aporta un ~10% medible y la ventaja sobre el QQQ es
  dependiente del régimen, **no** alpha sistemático.
- **Palancas descartadas**: HRP (Hierarchical Risk Parity) y arbitraje estadístico
  de pares se implementaron y midieron con el mismo protocolo; ninguna mejora el
  resultado y su código se revirtió (Tier 11, `RESULTADOS_HONESTOS.md` §11).

```bash
# Reproducir el bot ML+LLM con noticias point-in-time
PYTHONPATH=. python tools/build_news_sentiment.py        # FinBERT puntúa 61k titulares NASDAQ
PYTHONPATH=. python tools/ml_llm_news_backtest.py --mode balanced --n-windows 10 --seed 42
```

---

## 🔬 Basado en Investigación

El sistema está basado en papers de investigación de vanguardia:

1. **QuantAgents** (EMNLP 2025) - Multi-agent financial system [2]
2. **HedgeAgents** - Balanced-aware multi-agent trading [3]
3. **N-BEATS** - Neural basis expansion for time series [4]
4. **GARCH** - Volatility clustering models [21]

---

## 🛠️ Tecnologías

| Tecnología | Propósito |
|------------|-----------|
| Python 3.13 | Lenguaje principal |
| Flask | API REST + frontend |
| pandas/numpy | Manipulación de datos |
| scikit-learn | Machine Learning |
| xgboost/lightgbm | Gradient Boosting |
| arch | Modelos GARCH |
| yfinance | Datos financieros |
| feedparser | RSS feeds |
| transformers + FinBERT | Sentimiento financiero finetuneado |
| Ollama / llama-cpp-python | LLM local |
| HuggingFace | LLM API gratuita |

---

## 📄 Licencia

Este proyecto es de uso académico para el Trabajo Fin de Grado.

---

## 👨‍💻 Autor

**Luis Azofra Begara** ([@LuisAzofra](https://github.com/LuisAzofra)) — Trabajo Fin de Grado,
Grado en Ingeniería Informática, ETSI Informáticos, Universidad Politécnica de Madrid.

---

## 🙏 Agradecimientos

- Universidad Politécnica de Madrid
- Tutores académicos
- Comunidad de código abierto
- Autores de papers de investigación citados
