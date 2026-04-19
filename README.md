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

### ✅ LLM Gratuito
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
tfg_finance_ai/
├── data/                       # Ingesta de datos
│   ├── data_loader.py         # Yahoo Finance
│   └── news_fetcher.py        # RSS feeds reales
├── models/
│   ├── ml_models/
│   │   ├── traditional_ml.py  # RF, XGBoost, LightGBM
│   │   ├── deep_learning.py   # LSTM, GRU
│   │   └── volatility_models.py # GARCH, EGARCH
│   └── pattern_detection/
│       └── chart_patterns.py  # Patrones chartistas
├── agents/
│   ├── base_agent.py
│   ├── technical_agent/       # Análisis técnico
│   ├── sentiment_agent/       # LLM + noticias reales
│   ├── risk_agent/            # GARCH + VaR
│   └── manager_agent/         # LLM intérprete
├── evaluation/
│   └── backtest.py            # Métricas económicas
├── utils/
│   ├── data_utils.py
│   └── llm_client.py          # Ollama/HuggingFace
├── memoria/
│   ├── ESTADO_DEL_ARTE.md     # Estado del arte completo
│   └── MEMORIA_TFG.md         # Memoria del TFG
├── main.py                    # Script principal
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone <repositorio>
cd tfg_finance_ai
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
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

# Descargar modelo
ollama pull llama3.2
```

### 5. Verificar instalación

```bash
python -c "import arch; import yfinance; print('✅ Todo instalado correctamente')"
```

---

## 📊 Uso

### Ejecutar el sistema completo

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

### Opción 1: Ollama (Recomendado, gratuito)

```python
# El sistema detecta automáticamente Ollama
# Asegúrate de que esté corriendo:
ollama serve

# En otra terminal:
python main.py
```

### Opción 2: Hugging Face (Gratuito con rate limits)

```python
# No requiere configuración adicional
# El sistema usa Hugging Face como fallback automático
```

### Opción 3: Sin LLM (Fallback a reglas)

```python
# Si no hay LLM disponible, el sistema usa
# análisis basado en reglas (VADER, palabras clave)
```

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

---

## 🧪 Testing

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

El sistema produce los siguientes outputs:

```
results/
├── REPORTE_FINAL_AAPL.txt     # Reporte detallado
├── REPORTE_FINAL_MSFT.txt
├── backtest_analysis.png      # Gráficos de backtest
└── price_chart.png            # Gráficos de precios
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
| Python 3.9+ | Lenguaje principal |
| pandas/numpy | Manipulación de datos |
| scikit-learn | Machine Learning |
| xgboost/lightgbm | Gradient Boosting |
| arch | Modelos GARCH |
| yfinance | Datos financieros |
| feedparser | RSS feeds |
| Ollama | LLM local |
| HuggingFace | LLM API gratuita |

---

## 📄 Licencia

Este proyecto es de uso académico para el Trabajo Fin de Grado.

---

## 👨‍💻 Autor

Trabajo Fin de Grado

---

## 🙏 Agradecimientos

- Universidad Politécnica de Madrid
- Tutores académicos
- Comunidad de código abierto
- Autores de papers de investigación citados
