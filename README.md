# A Prospective Study of LLMs and ML for Financial Prediction

## Bachelor's Thesis (TFG) - Version 2.0

**Hybrid System with Real Data, Free LLMs and GARCH**

This project implements a complete hybrid system that combines traditional Machine Learning models, GARCH models for volatility forecasting, and a multi-agent system built on free LLMs (Ollama/Hugging Face) to predict financial assets and cryptoassets.

> The thesis itself (`memoria/`) and the experiment logs are written in Spanish.

---

## 🎯 Key Features

### ✅ Real Data (No Mocks)
- **Historical prices**: Yahoo Finance API
- **Financial news**: RSS feeds from real sources (Yahoo Finance, MarketWatch, CoinDesk)
- **Sentiment indicators**: Fear & Greed Index

### ✅ Free LLM / NLP
- **FinBERT** (`ProsusAI/finbert`): Transformer model **fine-tuned** on financial text, the main provider of the sentiment agent (see `utils/finbert_sentiment.py`)
- **Ollama**: local models (default `qwen2.5`)
- **Hugging Face**: inference API
- **Fallback**: rule-based analysis when no LLM is available

### ✅ Mathematical Volatility Forecasting
- **GARCH(1,1)**: statistically grounded model
- **EGARCH**: captures asymmetric effects
- **Forecasts with confidence intervals**

### ✅ Multi-Agent System
- **TechnicalAnalystAgent**: technical indicators
- **SentimentAnalystAgent**: sentiment analysis on real news
- **RiskManagerAgent**: VaR, CVaR, GARCH, position sizing
- **PortfolioManagerAgent**: LLM as interpreter

### ✅ Rigorous Economic Backtesting
- **Professional metrics**: Sharpe, Sortino, Calmar, Ulcer Index
- **Robustness analysis across market regimes**
- **Parameter sensitivity testing**

---

## 📁 Project Structure

```
.
├── api.py                      # Flask server: REST API + serves the frontend
├── frontend/                   # Web UI (HTML/CSS/JS)
├── main.py                     # Full console analysis (writes results/)
├── data/                       # Data ingestion
│   ├── data_loader.py          # Yahoo Finance
│   ├── news_fetcher.py         # Real RSS feeds
│   ├── news_sentiment_nasdaq.csv  # Daily FinBERT sentiment (point-in-time)
│   └── universe_snapshots/     # S&P 500 constituents (survivorship test)
├── models/
│   ├── ml_models/
│   │   ├── traditional_ml.py   # RF, XGBoost, LightGBM
│   │   ├── deep_learning.py    # LSTM, GRU
│   │   ├── volatility_models.py # GARCH, EGARCH
│   │   └── meta_labeling.py    # Meta-labeling (López de Prado)
│   └── pattern_detection/
│       └── chart_patterns.py   # Chart patterns
├── agents/
│   ├── base_agent.py
│   ├── debate.py               # Bull/bear debate + LLM judge
│   ├── technical_agent/        # Technical analysis
│   ├── sentiment_agent/        # FinBERT/LLM + real news
│   ├── risk_agent/             # GARCH + VaR
│   └── manager_agent/          # LLM interpreter
├── hybrid_system/              # ML + LLM integration
├── trading_bot/                # Paper-trading bot + risk gate
├── evaluation/
│   └── backtest.py             # Economic metrics and robustness
├── utils/                      # LLM client, FinBERT, caches, metrics, calibration
├── tools/                      # Reproducible experiments, sweeps and dashboards
├── compare_*.py                # A/B comparisons against the local API
├── tests/                      # Causality (no look-ahead) and meta-labeling tests
├── notebooks/                  # Analysis notebook
├── memoria/                    # Final thesis (PDF), state of the art and honest results
├── PROGRESS.md                 # Log of every experiment (Tiers 1-11)
└── requirements.txt
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/LuisAzofra/LLM-ML-Financial-Prediction.git
cd LLM-ML-Financial-Prediction
```

### 2. Create a virtual environment

Developed and tested with Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama (optional, for a local LLM)

```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from https://ollama.com/download

# Pull the model (the system uses qwen2.5 by default)
ollama pull qwen2.5
```

### 5. Verify the installation

```bash
python -c "import arch; import yfinance; print('✅ All set')"
```

---

## 📊 Usage

### Web UI (API + frontend)

```bash
python api.py              # http://localhost:5000
PORT=5057 python api.py    # port used by the compare_*.py and tools/ scripts
```

### Full console analysis

```bash
python main.py
```

### Analysis notebook

```bash
jupyter notebook notebooks/analisis_completo.ipynb
```

### Programmatic example

```python
from data.data_loader import FinancialDataLoader
from data.news_fetcher import NewsFetcher
from models.ml_models.volatility_models import GARCHVolatilityModel
from utils.llm_client import LLMClient

# Load data
loader = FinancialDataLoader()
df = loader.download_stock_data('AAPL', '2020-01-01', '2024-01-01')

# Fetch real news
fetcher = NewsFetcher()
news = fetcher.fetch_yahoo_finance_news('AAPL', max_items=10)

# Forecast volatility with GARCH
returns = df['Close'].pct_change().dropna().values
garch = GARCHVolatilityModel(p=1, q=1)
garch.fit(returns)
forecast = garch.forecast(horizon=5)
print(f"Forecast volatility: {forecast.forecast_value*100:.2f}%")

# Sentiment analysis with an LLM
llm = LLMClient(provider="ollama")
result = llm.analyze_sentiment("Apple reports record earnings...")
print(f"Sentiment: {result['parsed']['sentiment']}")
```

---

## 🔧 LLM Configuration

The provider is selected with the `TFG_LLM_PROVIDER` environment variable:

| Provider | Default model | Notes |
|----------|---------------|-------|
| `ollama` (API default) | `qwen2.5` | Requires `ollama serve` on `localhost:11434` |
| `local` | `Qwen/Qwen2.5-0.5B-Instruct` | Local `transformers`; override with `TFG_LOCAL_LLM` |
| `local-gguf` | Qwen2.5-1.5B-Instruct Q4 (GGUF) | Requires `llama-cpp-python`; file set by `TFG_LOCAL_GGUF_FILE` |
| `huggingface` | `mistralai/Mistral-7B-Instruct-v0.2` | Hugging Face inference API; override with `TFG_HF_MODEL` |

```bash
ollama serve                                   # in one terminal
TFG_LLM_PROVIDER=ollama python api.py          # in another
```

Other useful variables:

- `FINBERT_MODEL` (default `ProsusAI/finbert`): financial sentiment model.
- `TFG_LIGHTWEIGHT=1` / `TFG_DISABLE_TF=1`: lightweight start without TensorFlow (no LSTM/GRU).
- `PRICE_CACHE=1`: use the local price cache (useful for reproducible tests).

If no LLM is available, the system falls back to rule-based analysis (VADER, keywords).

---

## 📈 Implemented Metrics

### Performance
- Total Return
- Annualized Return

### Risk
- Annualized Volatility
- Maximum Drawdown
- Ulcer Index
- VaR and CVaR

### Risk-Adjusted
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

## 📚 Thesis Documents (Spanish)

- **[Final thesis (PDF)](memoria/Memoria_TFG.pdf)**: version submitted to the ETSI Informáticos (June 2026), including an appendix on prompt design and agent context management. The FinBERT experiments, the ML+LLM news bot and the HRP/pairs levers (Tiers 9-11) are documented in the files below
- **[State of the art](memoria/ESTADO_DEL_ARTE.md)**: review of the 2024-2026 scientific literature
- **[Thesis (Markdown)](memoria/MEMORIA_TFG.md)**: design, implementation and results
- **[Honest results](memoria/RESULTADOS_HONESTOS.md)**: evaluation without look-ahead, out-of-sample and lockbox
- **[ML+LLM news bot](memoria/RESULTADOS_ML_LLM_NOTICIAS.md)**: point-in-time FinBERT + RandomForest
- **[PROGRESS](PROGRESS.md)**: log of every experiment, accepted or rejected

---

## 🧪 Testing

### Automated tests

They check that there is no look-ahead (invariance when the history is truncated) and validate the meta-labeling:

```bash
PYTHONPATH=. python tests/test_causality.py
PYTHONPATH=. python tests/test_meta_labeling.py
```

### Robustness tests

```python
from evaluation.backtest import Backtester, RobustnessTester

backtester = Backtester()
robustness_tester = RobustnessTester(backtester)

# Per-regime test
regime_results = robustness_tester.test_regime_robustness(
    df, signal_generator, 'AAPL'
)

# Robustness score
score = robustness_tester.calculate_robustness_score(regime_results)
print(f"Robustness score: {score['overall']:.2f}")
```

---

## 📊 Outputs

`main.py` produces:

```
results/
├── REPORTE_FINAL_<SYMBOL>.txt   # Detailed report per asset
├── backtest_analysis.png        # Backtest charts
└── price_chart.png              # Price charts
```

The files committed in `results/` and `RESUMEN_EJECUCION.txt` are a sample run of the
first version (February 2026). The current, rigorous results are in
`memoria/RESULTADOS_HONESTOS.md`.

---

## 📈 Real Results (Honest Evaluation)

Rigorous evaluation (no look-ahead, out-of-sample, *Deflated Sharpe*). Full details in
[`memoria/RESULTADOS_HONESTOS.md`](memoria/RESULTADOS_HONESTOS.md),
[`memoria/RESULTADOS_ML_LLM_NOTICIAS.md`](memoria/RESULTADOS_ML_LLM_NOTICIAS.md) and
[`PROGRESS.md`](PROGRESS.md) (Tiers 7-11).

- **Main finding**: the added sophistication (ML + LLM + multi-agent) **does not
  generate alpha** over passive indexing on a risk-adjusted basis, consistent with
  the efficient market hypothesis. A valid and defensible negative result.
- **ML+LLM bot with real *point-in-time* news (FinBERT + RandomForest)** →
  **positive *out-of-sample* returns**. Trained on 2019-2020, tested on 2021-2023:

  | Horizon | ML+LLM (mean) | % positive windows | QQQ buy & hold |
  |---------|--------------:|:------------------:|---------------:|
  | 1 month  | +0.04% | 62% | −0.30% |
  | 6 months | +7.33% | 62% | +5.50% |
  | 1 year   | +7.72% | 62% | +4.68% |
  | 2 years  | +5.72% | 75% | +0.84% |

  Overall OOS **+5.2% mean, 66% positive windows** (stable across 5 seeds).
  Honestly: most of the return comes from disciplined *long/flat* exposure to the
  index; FinBERT (LLM) contributes a measurable ~10%, and the edge over QQQ is
  regime-dependent, **not** systematic alpha.
- **Rejected levers**: HRP (Hierarchical Risk Parity) and statistical pairs
  arbitrage were implemented and measured with the same protocol; neither improves
  the result and their code was reverted (Tier 11, `RESULTADOS_HONESTOS.md` §11).

```bash
# Reproduce the ML+LLM bot with point-in-time news
PYTHONPATH=. python tools/build_news_sentiment.py        # FinBERT scores 61k NASDAQ headlines
PYTHONPATH=. python tools/ml_llm_news_backtest.py --mode balanced --n-windows 10 --seed 42
```

---

## 🔬 Research Background

The system builds on recent research:

1. **QuantAgents** (EMNLP 2025) - Multi-agent financial system
2. **HedgeAgents** (WWW 2025) - Balanced-aware multi-agent trading
3. **N-BEATS** (ICLR 2020) - Neural basis expansion for time series
4. **GARCH** (Engle, 1982) - Volatility clustering models

---

## 🛠️ Technologies

| Technology | Purpose |
|------------|---------|
| Python 3.13 | Main language |
| Flask | REST API + frontend |
| pandas/numpy | Data manipulation |
| scikit-learn | Machine Learning |
| xgboost/lightgbm | Gradient Boosting |
| arch | GARCH models |
| yfinance | Financial data |
| feedparser | RSS feeds |
| transformers + FinBERT | Fine-tuned financial sentiment |
| Ollama / llama-cpp-python | Local LLM |
| Hugging Face | LLM inference API |

---

## 📄 License

This project is for academic use as part of a Bachelor's Thesis.

---

## 👨‍💻 Author

**Luis Azofra Begara** ([@LuisAzofra](https://github.com/LuisAzofra)) — Bachelor's Thesis,
BSc in Computer Engineering, ETSI Informáticos, Universidad Politécnica de Madrid.

---

## 🙏 Acknowledgements

- Universidad Politécnica de Madrid
- Academic supervisors
- The open-source community
- The authors of the cited research papers
