/* ═══════════════════════════════════════════
   SOVEREIGN ANALYST — Frontend Logic
   ═══════════════════════════════════════════ */

const API = '';  // same origin

// ── Global State ────────────────────────────
let currentType     = 'stock';
let currentTf       = '1y';
let currentView     = 'analysis';
let suggestionsData = {};
let priceChart      = null;
let donutChart      = null;
let btChart         = null;
let lastAnalysisData = null;   // stored for "How was this calculated?" modal
let lastBacktestData = null;   // stored for backtest explain modal

// ── IDs shown only in analysis view ─────────
const ANALYSIS_IDS = ['hero-heading','controls-bar','quick-picks'];
// ── IDs shown only in backtest view ─────────
const BACKTEST_IDS = ['backtest-section'];

// ── Timeframe mapping (frontend label → backend key) ─
const TF_MAP = {
  '5M': '1m', '1H': '1m', '12H': '3m',
  '24H': '1y', '1W': '1y', '1M': '2y', '1Y': '3y'
};

// ── Loading steps ───────────────────────────
const LOADING_STEPS = [
  'Downloading market data...',
  'Engineering 55+ technical features...',
  'Training ML models (RF, XGBoost, LightGBM)...',
  'Fitting GARCH volatility model...',
  'Running multi-agent analysis...',
  'Computing hybrid ML + LLM score...',
];

// ══════════════════════════════════════════════
//  INITIALIZATION
// ══════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('footer-year').textContent = new Date().getFullYear();
  setupNavigation();
  setupAssetToggle();
  setupTfPills();
  setupQuickPicks();
  setupSymbolInput();
  document.getElementById('analyze-btn').addEventListener('click', runAnalysis);
  document.getElementById('symbol-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') runAnalysis();
  });
  loadSuggestions();
  setupBacktest();
  setupBotView();
  setupCalcModal();
  setupTooltips();
  checkOllamaStatus();           // Check on load
  setInterval(checkOllamaStatus, 30000); // Re-check every 30s
});

// ══════════════════════════════════════════════
//  OLLAMA STATUS INDICATOR
// ══════════════════════════════════════════════
async function checkOllamaStatus() {
  const badge = document.getElementById('ollama-status-badge');
  if (!badge) return;
  try {
    const res  = await fetch(`${API}/api/ollama-status`);
    const data = await res.json();
    if (data.status === 'running') {
      badge.textContent = `🟢 LLM: ${data.active_model || 'OK'}`;
      badge.style.color = '#22c55e';
      badge.title = `Ollama corriendo · ${data.model_count} modelos disponibles: ${data.models.join(', ')}`;
    } else {
      badge.textContent = '🔴 LLM: Offline';
      badge.style.color = '#ef4444';
      badge.title = 'Ollama no está corriendo. Ejecuta: ollama serve\nEl análisis usará reglas heurísticas sin LLM.';
    }
  } catch {
    badge.textContent = '🔴 LLM: Offline';
    badge.style.color = '#ef4444';
    badge.title = 'No se puede contactar con el servidor Flask. ¿Está corriendo api.py?';
  }
}

// ══════════════════════════════════════════════
//  NAVIGATION / VIEW SWITCHING
// ══════════════════════════════════════════════
function setupNavigation() {
  document.querySelectorAll('.nav-link[data-view]').forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      switchView(link.dataset.view);
    });
  });
}

function switchView(view) {
  currentView = view;
  const isAnalysis = view === 'analysis';
  const isBacktest = view === 'backtest';
  const isBot      = view === 'bot';

  // Update active nav link
  document.querySelectorAll('.nav-link[data-view]').forEach(a => {
    a.classList.toggle('active', a.dataset.view === view);
  });

  // Show/hide analysis-only elements
  ANALYSIS_IDS.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = isAnalysis ? '' : 'none';
  });

  // Loading/error/results only visible in analysis view
  if (!isAnalysis) {
    ['loading-state', 'error-state', 'analysis-section'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.style.display = 'none';
    });
  }

  // Backtest section
  const bt = document.getElementById('backtest-section');
  if (bt) bt.style.display = isBacktest ? 'block' : 'none';

  // Bot section
  const botSec = document.getElementById('bot-section');
  if (botSec) {
    botSec.style.display = isBot ? 'block' : 'none';
    if (isBot) botLoadStatus(false);   // load without fetching live prices (fast)
  }

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ══════════════════════════════════════════════
//  TOOLTIP SYSTEM (JS floating, not CSS ::after)
// ══════════════════════════════════════════════
function setupTooltips() {
  const tip = document.createElement('div');
  tip.id = 'js-tooltip';
  document.body.appendChild(tip);

  let activeIcon = null;

  document.addEventListener('mouseover', e => {
    const icon = e.target.closest('.info-icon');
    if (!icon || !icon.dataset.tip) return;
    activeIcon = icon;
    tip.textContent = icon.dataset.tip;
    tip.style.opacity = '1';
    positionTooltip(tip, icon);
  });

  document.addEventListener('mouseout', e => {
    const icon = e.target.closest('.info-icon');
    if (icon) {
      tip.style.opacity = '0';
      activeIcon = null;
    }
  });

  // Reposition on scroll/resize
  document.addEventListener('scroll', () => {
    if (activeIcon) positionTooltip(tip, activeIcon);
  }, true);
}

function positionTooltip(tip, icon) {
  const rect = icon.getBoundingClientRect();
  const tipW = 240;
  let left = rect.left + rect.width / 2 - tipW / 2;
  let top  = rect.top - 8;  // will subtract tip height below

  // We need layout to get height — approximate as 60px then adjust
  left = Math.max(8, Math.min(left, window.innerWidth - tipW - 8));
  top  = rect.top - 68;  // above icon
  if (top < 8) top = rect.bottom + 8;  // flip below if no space

  tip.style.left = left + 'px';
  tip.style.top  = top + 'px';
}

// ══════════════════════════════════════════════
//  ASSET TYPE TOGGLE
// ══════════════════════════════════════════════
function setupAssetToggle() {
  document.querySelectorAll('.asset-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.asset-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentType = btn.dataset.type;
      document.getElementById('symbol-input').placeholder =
        currentType === 'crypto' ? 'Search BTC, ETH, SOL...' : 'Search AAPL, MSFT, NVDA...';
      document.getElementById('symbol-input').value = '';
    });
  });
}

// ══════════════════════════════════════════════
//  TIMEFRAME PILLS
// ══════════════════════════════════════════════
function setupTfPills() {
  document.querySelectorAll('.tf-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tf-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentTf = btn.dataset.tf;
    });
  });
}

// ══════════════════════════════════════════════
//  QUICK PICKS
// ══════════════════════════════════════════════
function setupQuickPicks() {
  document.querySelectorAll('.pick-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const sym  = btn.dataset.symbol;
      const type = btn.dataset.type;
      currentType = type;
      document.querySelectorAll('.asset-pill').forEach(b => {
        b.classList.toggle('active', b.dataset.type === type);
      });
      document.getElementById('symbol-input').value = sym;
      // Do NOT auto-run — user must click Analyze
    });
  });
}

// ══════════════════════════════════════════════
//  SYMBOL INPUT + AUTOCOMPLETE
// ══════════════════════════════════════════════
async function loadSuggestions() {
  try {
    const r = await fetch(`${API}/api/suggestions`);
    suggestionsData = await r.json();
  } catch(_) {
    suggestionsData = {
      stocks: ['AAPL','MSFT','GOOGL','AMZN','TSLA','META','NVDA','JPM'],
      crypto: ['BTC-USD','ETH-USD','BNB-USD','SOL-USD','XRP-USD']
    };
  }
}

function setupSymbolInput() {
  const inp  = document.getElementById('symbol-input');
  const drop = document.getElementById('suggestions-dropdown');

  inp.addEventListener('input', () => {
    const val  = inp.value.trim().toUpperCase();
    const pool = currentType === 'crypto' ? suggestionsData.crypto : suggestionsData.stocks;
    if (!pool || !val) { drop.style.display = 'none'; return; }

    const matches = pool.filter(s => s.includes(val)).slice(0, 8);
    if (!matches.length) { drop.style.display = 'none'; return; }

    drop.innerHTML = matches.map(s =>
      `<div class="suggestion-item">${s}</div>`
    ).join('');
    drop.style.display = 'block';

    drop.querySelectorAll('.suggestion-item').forEach(item => {
      item.addEventListener('click', () => {
        inp.value = item.textContent;
        drop.style.display = 'none';
        // Do NOT auto-run — user must click Analyze
      });
    });
  });

  document.addEventListener('click', e => {
    if (!inp.contains(e.target) && !drop.contains(e.target))
      drop.style.display = 'none';
  });
  inp.addEventListener('keydown', e => {
    if (e.key === 'Escape') drop.style.display = 'none';
  });
}

// ══════════════════════════════════════════════
//  MAIN ANALYSIS
// ══════════════════════════════════════════════
async function runAnalysis() {
  const symbol = document.getElementById('symbol-input').value.trim().toUpperCase();
  if (!symbol) {
    document.getElementById('symbol-input').focus();
    return;
  }

  // Make sure we're in analysis view
  if (currentView !== 'analysis') switchView('analysis');

  showLoading(true);
  hideResults();
  hideError();

  // Simulate progress steps
  let step = 0;
  const stepInterval = setInterval(() => {
    if (step < LOADING_STEPS.length) {
      document.getElementById('loading-text').textContent = LOADING_STEPS[step] || 'Processing...';
      document.getElementById('loading-sub').textContent = `Step ${step + 1}/${LOADING_STEPS.length}`;
      document.getElementById('progress-fill').style.width = `${((step + 1) / LOADING_STEPS.length) * 90}%`;
      step++;
    }
  }, 3200);

  try {
    const res = await fetch(`${API}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, timeframe: currentTf, asset_type: currentType }),
    });
    const data = await res.json();

    clearInterval(stepInterval);
    document.getElementById('progress-fill').style.width = '100%';
    await sleep(300);

    showLoading(false);

    if (data.status === 'error') {
      showError(data.error || 'Unknown error from API');
      return;
    }

    renderResults(data);

  } catch (err) {
    clearInterval(stepInterval);
    showLoading(false);
    showError(`Connection failed: ${err.message}. Is the Flask server running?`);
  }
}

// ══════════════════════════════════════════════
//  RENDER ALL RESULTS
// ══════════════════════════════════════════════
function renderResults(data) {
  lastAnalysisData = data;
  renderTickerHeader(data);
  renderProjectionBanner(data);
  renderConsensusSignal(data);
  renderVolatility(data);
  renderPriceChart(data);
  renderMLTable(data);
  renderAgentCards(data);
  document.getElementById('analysis-section').style.display = 'block';
  document.getElementById('analysis-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Ticker Header ────────────────────────────
function renderTickerHeader(data) {
  const info   = data.data_info || {};
  const hybrid = data.hybrid    || {};
  const symbol = data.symbol    || '—';
  const price  = info.current_price;
  const type   = data.asset_type;

  document.getElementById('ticker-symbol').textContent = symbol;
  document.getElementById('ticker-price').textContent  = price ? '$' + fmtNum(price) : '—';

  const score = hybrid.score || 0;
  const pctChange = data.ml?.ensemble_prediction ? (data.ml.ensemble_prediction * 100).toFixed(2) : null;
  const badge = document.getElementById('price-change-badge');
  if (pctChange !== null) {
    const isPos = parseFloat(pctChange) >= 0;
    badge.textContent = (isPos ? '+' : '') + pctChange + '%';
    badge.className = 'price-change-badge ' + (isPos ? 'positive' : 'negative');
  } else {
    badge.textContent = '—'; badge.className = 'price-change-badge neutral';
  }

  const now = new Date();
  document.getElementById('last-update').textContent = `Last Update: ${now.toLocaleTimeString('es-ES')}`;
  document.getElementById('ticker-sub').textContent  =
    `${symbol} · ${type === 'crypto' ? 'Crypto' : 'Equity'} · ${data.timeframe?.toUpperCase() || ''} Analysis · ${info.records || 0} data points`;
}

// ── Projected Price Banner ────────────────────
function renderProjectionBanner(data) {
  const info    = data.data_info || {};
  const ml      = data.ml        || {};
  const price   = info.current_price;
  const ensemblePred = ml.ensemble_prediction;
  const horizonDays  = ml.prediction_horizon_days || 5;

  if (!price || ensemblePred === undefined || ensemblePred === null) {
    document.getElementById('projection-banner').style.display = 'none';
    return;
  }

  const target  = price * (1 + ensemblePred);
  const retPct  = (ensemblePred * 100);
  const isPos   = retPct >= 0;
  const conf    = ml.ensemble_confidence || 0;
  const confPct = Math.round(conf * 100);

  // Horizon label
  const horizonLabel = horizonDays <= 7 ? `${horizonDays}-day forecast`
    : horizonDays <= 14 ? '2-week forecast'
    : horizonDays <= 31 ? '1-month forecast'
    : `${horizonDays}-day forecast`;

  const confLabel = confPct >= 70 ? 'High confidence' : confPct >= 50 ? 'Moderate confidence' : 'Low confidence';

  document.getElementById('proj-current').textContent = '$' + fmtNum(price);
  document.getElementById('proj-target').textContent  = '$' + fmtNum(target);
  document.getElementById('proj-horizon').textContent = horizonLabel + ' · ' + confLabel;

  const retEl = document.getElementById('proj-return');
  retEl.textContent = (isPos ? '+' : '') + retPct.toFixed(2) + '%';
  retEl.className   = 'proj-return-val ' + (isPos ? 'positive' : 'negative');

  document.getElementById('proj-conf').textContent = confPct + '% model confidence';

  document.getElementById('projection-banner').style.display = 'flex';
}

// ── Consensus Signal (Donut Gauge) ───────────
function renderConsensusSignal(data) {
  const hybrid = data.hybrid || {};
  const score  = hybrid.score || 0;
  const conf   = hybrid.confidence || 0;
  const rec    = hybrid.recommendation || 'MANTENER';

  // Color based on recommendation
  const color = recColor(rec);

  // Donut gauge
  const pct = Math.round(((score + 1) / 2) * 100);  // map [-1,1] → [0,100]
  const remaining = 100 - pct;

  if (donutChart) donutChart.destroy();
  donutChart = new Chart(document.getElementById('donut-gauge'), {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [pct, remaining],
        backgroundColor: [color, 'rgba(255,255,255,0.04)'],
        borderWidth: 0,
        cutout: '78%',
      }]
    },
    options: {
      animation: { animateRotate: true, duration: 900 },
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      responsive: true, maintainAspectRatio: true,
    }
  });

  document.getElementById('donut-gauge').style.position = 'absolute';
  document.getElementById('donut-gauge').style.top = '0';
  document.getElementById('donut-gauge').style.left = '0';

  const recText = translateRec(rec);
  document.getElementById('consensus-rec-text').textContent = recText;
  document.getElementById('consensus-rec-text').style.color = color;

  // Intensity badge
  const absScore = Math.abs(score);
  const intensity = absScore > 0.6 ? 'HIGH INTENSITY' : absScore > 0.3 ? 'MEDIUM INTENSITY' : 'LOW INTENSITY';
  document.getElementById('intensity-badge').textContent = intensity;

  // Confidence bar
  const confPct = Math.round(conf * 100);
  document.getElementById('conf-fill').style.width = confPct + '%';
  document.getElementById('conf-pct').textContent = confPct + '%';

  // Quote
  const quotes = {
    COMPRA_FUERTE: '"Strong converging signals support an aggressive long position."',
    COMPRA:        '"Our analysis indicates a buy signal based on momentum convergence."',
    MANTENER:      '"Mixed signals suggest a cautious hold with tight risk controls."',
    VENTA:         '"Deteriorating momentum warrants reduced exposure."',
    VENTA_FUERTE:  '"Strong sell signal — risk management takes priority."',
  };

  // Direction conflict notice
  const conflict = data.hybrid?.direction_conflict;
  const mlDir    = (data.ml?.ensemble_prediction ?? 0) >= 0 ? 'upward' : 'downward';
  const llmRec   = translateRec(data.agents?.individual?.ml_prediction?.recommendation || rec);
  const conflictNote = conflict
    ? `<div class="conflict-notice">⚠ ML projects <strong>${mlDir}</strong> but LLM agents lean ${llmRec === 'COMPRA' ? 'bullish' : 'bearish'} — signals conflict, holding is prudent.</div>`
    : '';

  document.getElementById('analysis-quote').innerHTML =
    `<em>${quotes[rec] || quotes.MANTENER}</em>${conflictNote}`;
}

// ── Market Volatility ─────────────────────────
function renderVolatility(data) {
  const garch = data.garch;
  const info  = data.data_info || {};

  let volPct  = info.annualized_volatility || null;
  let volStr  = volPct ? volPct.toFixed(1) + '%' : '—';
  document.getElementById('vol-pct').textContent = volStr;

  let riskClass = 'risk-medium', riskLabel = 'Moderate Risk';
  if (volPct !== null) {
    if (volPct < 15)       { riskClass = 'risk-low';     riskLabel = 'Low Risk'; }
    else if (volPct < 30)  { riskClass = 'risk-medium';  riskLabel = 'Moderate Risk'; }
    else if (volPct < 60)  { riskClass = 'risk-high';    riskLabel = 'Elevated Risk'; }
    else                   { riskClass = 'risk-extreme';  riskLabel = 'Extreme Risk'; }
  }
  const riskBadge = document.getElementById('risk-badge');
  riskBadge.textContent = riskLabel;
  riskBadge.className = 'risk-badge ' + riskClass;

  // Mini bar chart (7 bars, random heights for visualization)
  const chart = document.getElementById('vol-mini-chart');
  chart.innerHTML = '';
  const heights = [35,45,30,55,40,70,60];
  heights.forEach((h, i) => {
    const bar = document.createElement('div');
    bar.className = 'vol-bar' + (i === heights.length - 1 ? ' active' : '');
    bar.style.height = h + '%';
    chart.appendChild(bar);
  });

  // GARCH meta
  const meta = document.getElementById('vol-meta');
  if (garch) {
    meta.textContent = `GARCH forecast: ${garch.forecast_volatility?.toFixed(2)}% · CI [${garch.ci_lower?.toFixed(2)}, ${garch.ci_upper?.toFixed(2)}]`;
  } else {
    meta.textContent = '';
  }
}

// ── Price Chart ──────────────────────────────
function renderPriceChart(data) {
  const pc = data.price_chart || {};
  const dates = pc.dates || [];
  const close = pc.close || [];
  const sma20 = pc.sma_20 || [];
  const sma50 = pc.sma_50 || [];

  if (!dates.length) return;

  if (priceChart) priceChart.destroy();
  const ctx = document.getElementById('price-chart').getContext('2d');

  priceChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: dates,
      datasets: [
        {
          label: 'Close Price',
          data: close,
          backgroundColor: close.map((v, i) => {
            if (i === 0) return 'rgba(13,207,207,0.5)';
            return v >= close[i-1] ? 'rgba(16,185,129,0.6)' : 'rgba(239,68,68,0.6)';
          }),
          borderColor: close.map((v, i) => {
            if (i === 0) return 'rgba(13,207,207,0.8)';
            return v >= close[i-1] ? '#10b981' : '#ef4444';
          }),
          borderWidth: 1,
          borderRadius: 2,
          type: 'bar',
          order: 2,
        },
        {
          label: 'SMA 20',
          data: sma20,
          borderColor: '#10b981',
          borderWidth: 1.5,
          borderDash: [5, 3],
          pointRadius: 0,
          fill: false,
          type: 'line',
          order: 1,
        },
        {
          label: 'SMA 50',
          data: sma50,
          borderColor: '#f59e0b',
          borderWidth: 1.5,
          borderDash: [5, 3],
          pointRadius: 0,
          fill: false,
          type: 'line',
          order: 0,
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: true,
          labels: { color: '#94a3b8', font: { size: 11 }, boxWidth: 16, usePointStyle: true },
        },
        tooltip: {
          backgroundColor: 'rgba(15,20,33,0.95)',
          borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1,
          titleColor: '#f1f5f9', bodyColor: '#94a3b8',
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: $${fmtNum(ctx.parsed.y)}`,
          }
        }
      },
      scales: {
        x: {
          ticks: {
            color: '#475569', maxTicksLimit: 10, font: { size: 10 },
            maxRotation: 0,
          },
          grid: { color: 'rgba(255,255,255,0.03)' }
        },
        y: {
          ticks: {
            color: '#475569', font: { size: 10 },
            callback: v => '$' + fmtNum(v),
          },
          grid: { color: 'rgba(255,255,255,0.04)' }
        }
      }
    }
  });

  // Set canvas height
  document.getElementById('price-chart').parentElement.style.height = '280px';
}

// ── ML Table ─────────────────────────────────
function renderMLTable(data) {
  const ml   = data.ml || {};
  const models = ml.models || [];
  const preds  = ml.predictions || {};
  const ensDir = ml.ensemble_direction || '—';
  const ensConf = ml.ensemble_confidence || 0;

  const MODEL_LABELS = {
    random_forest: 'Random Forest',
    xgboost:       'XGBoost',
    lightgbm:      'LightGBM',
    lstm:          'LSTM Recurrent',
  };

  const tbody = document.getElementById('ml-tbody');
  tbody.innerHTML = '';

  models.forEach(m => {
    const pred   = preds[m.name] || 0;
    const dir    = pred > 0.001 ? 'Long' : pred < -0.001 ? 'Short' : 'Neutral';
    const dirStr = pred > 0.002 ? 'Strong Long' : pred > 0.001 ? 'Long' : pred < -0.002 ? 'Strong Short' : pred < -0.001 ? 'Short' : 'Neutral';
    const dirCls = dir === 'Long' ? 'pred-long' : dir === 'Short' ? 'pred-short' : 'pred-neutral';
    const confPct = Math.round(Math.max(0.1, Math.min(1.0, (m.r2 + 1) / 2)) * 100);

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="font-weight:600;color:var(--text-primary)">${MODEL_LABELS[m.name] || m.name}</td>
      <td>${m.rmse?.toFixed(3) ?? '—'}</td>
      <td>${m.r2?.toFixed(3) ?? '—'}</td>
      <td class="${dirCls}">${dirStr}</td>
      <td>
        <span class="conf-mini-bar">
          <span class="conf-mini-fill" style="width:${confPct}%"></span>
        </span>
      </td>`;
    tbody.appendChild(tr);
  });

  const meta = document.getElementById('ml-meta');
  const lstmNote = ml.lstm_available ? ' + LSTM' : '';
  const optNote  = ml.stacking_used ? ' · Stacking meta-learner' : '';
  meta.textContent = [
    `Ensemble: ${(ensConf * 100).toFixed(1)}% confidence · ${ensDir}`,
    ml.n_features ? `${ml.n_features} features` : '',
    ml.prediction_horizon_days ? `${ml.prediction_horizon_days}-day horizon` : '',
    `RF + XGBoost + LightGBM${lstmNote}${optNote}`,
  ].filter(Boolean).join(' · ');
}

// ── Agent Cards ──────────────────────────────
function renderAgentCards(data) {
  const individual = data.agents?.individual || {};

  const AGENT_META = [
    {
      key: 'technical',
      icon: '📊',
      title: 'Technical Analysis',
      desc: 'Evaluating historical price patterns and multi-layer volume indicators.',
      dot: 'dot-teal',
    },
    {
      key: 'sentiment',
      icon: '💬',
      title: 'Sentiment Insights',
      desc: 'Parsing global social feeds and institutional news for market mood.',
      dot: 'dot-green',
    },
    {
      key: 'risk',
      icon: '🛡',
      title: 'Risk Management',
      desc: 'Stress-testing volatility thresholds and potential drawdown levels.',
      dot: 'dot-amber',
    },
    {
      key: 'ml_prediction',
      icon: '🤖',
      title: 'Machine Learning',
      desc: 'Stacking ensemble projections from 55+ financial indicators.',
      dot: 'dot-blue',
    },
  ];

  const STATUS_MAP = {
    COMPRA_FUERTE: { label: 'STRONG BUY',  dot: 'dot-teal' },
    COMPRA:        { label: 'ACTIVE SIGNAL', dot: 'dot-teal' },
    MANTENER:      { label: 'BULLISH LEAN', dot: 'dot-green' },
    VENTA:         { label: 'BEARISH LEAN', dot: 'dot-amber' },
    VENTA_FUERTE:  { label: 'SELL SIGNAL',  dot: 'dot-red' },
    RECHAZAR:      { label: 'GUARDRAILS UP', dot: 'dot-red' },
    APROBAR:       { label: 'APPROVED',     dot: 'dot-green' },
  };

  const grid = document.getElementById('agent-grid');
  grid.innerHTML = '';

  AGENT_META.forEach((meta, i) => {
    const agent = individual[meta.key] || {};
    const rec   = agent.recommendation || '—';
    const conf  = agent.confidence ? Math.round(agent.confidence * 100) : null;
    const status = STATUS_MAP[rec] || { label: rec !== '—' ? rec : 'ANALYZING', dot: meta.dot };

    const card = document.createElement('div');
    card.className = 'agent-card';
    card.style.animationDelay = `${i * 0.08}s`;
    card.innerHTML = `
      <div class="agent-icon">${meta.icon}</div>
      <div class="agent-title">${meta.title}</div>
      <div class="agent-desc">${meta.desc}</div>
      <div class="agent-status">
        <div class="status-dot ${status.dot}"></div>
        <span style="color:var(--text-secondary)">${status.label}</span>
      </div>
      ${conf !== null ? `<div class="agent-rec">Confidence: ${conf}%</div>` : ''}
      ${agent.reasoning ? `<div class="agent-rec" style="margin-top:0.5rem;font-style:italic;font-size:0.72rem">${agent.reasoning.slice(0, 120)}${agent.reasoning.length > 120 ? '…' : ''}</div>` : ''}
    `;
    grid.appendChild(card);
  });
}

// ══════════════════════════════════════════════
//  HISTORICAL BACKTEST
// ══════════════════════════════════════════════
function setupBacktest() {
  // Set max date to 30 days ago (need enough actual data to compare)
  const maxDate = new Date();
  maxDate.setDate(maxDate.getDate() - 30);
  const maxStr = maxDate.toISOString().split('T')[0];
  document.getElementById('bt-date').max = maxStr;
  // Default to 1 year ago
  const defaultDate = new Date();
  defaultDate.setFullYear(defaultDate.getFullYear() - 1);
  document.getElementById('bt-date').value = defaultDate.toISOString().split('T')[0];

  document.getElementById('bt-run-btn').addEventListener('click', runHistoricalBacktest);

  // Quick pick buttons
  document.querySelectorAll('.bt-pick-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const sym  = btn.dataset.symbol;
      const type = btn.dataset.type;
      document.getElementById('bt-symbol').value = sym;
      document.getElementById('bt-asset-type').value = type;
      document.querySelectorAll('.bt-pick-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('bt-suggestions-dropdown').style.display = 'none';
    });
  });

  // Autocomplete for bt-symbol
  const inp  = document.getElementById('bt-symbol');
  const drop = document.getElementById('bt-suggestions-dropdown');

  inp.addEventListener('input', () => {
    const val  = inp.value.trim().toUpperCase();
    const type = document.getElementById('bt-asset-type').value;
    const pool = type === 'crypto' ? suggestionsData.crypto : suggestionsData.stocks;
    if (!pool || !val) { drop.style.display = 'none'; return; }

    const matches = pool.filter(s => s.includes(val)).slice(0, 8);
    if (!matches.length) { drop.style.display = 'none'; return; }

    drop.innerHTML = matches.map(s =>
      `<div class="suggestion-item" data-sym="${s}">${s}</div>`
    ).join('');
    drop.style.display = 'block';

    drop.querySelectorAll('.suggestion-item').forEach(item => {
      item.addEventListener('mousedown', e => {
        e.preventDefault();
        inp.value = item.dataset.sym;
        // auto-detect type
        const isCrypto = suggestionsData.crypto && suggestionsData.crypto.includes(item.dataset.sym);
        document.getElementById('bt-asset-type').value = isCrypto ? 'crypto' : 'stock';
        drop.style.display = 'none';
        // sync active quick pick
        document.querySelectorAll('.bt-pick-btn').forEach(b => {
          b.classList.toggle('active', b.dataset.symbol === item.dataset.sym);
        });
      });
    });
  });

  inp.addEventListener('blur', () => { setTimeout(() => { drop.style.display = 'none'; }, 150); });
  inp.addEventListener('keydown', e => {
    if (e.key === 'Enter') { drop.style.display = 'none'; runHistoricalBacktest(); }
    if (e.key === 'Escape') { drop.style.display = 'none'; }
  });
}

async function runHistoricalBacktest() {
  const symbol   = document.getElementById('bt-symbol').value.trim().toUpperCase();
  const date     = document.getElementById('bt-date').value;
  const horizon  = document.getElementById('bt-horizon').value;
  const assetType= document.getElementById('bt-asset-type').value;

  if (!symbol) {
    document.getElementById('bt-symbol').focus(); return;
  }
  if (!date) {
    document.getElementById('bt-date').focus(); return;
  }

  document.getElementById('bt-loading').style.display = 'flex';
  document.getElementById('bt-results').style.display = 'none';
  document.getElementById('bt-error').style.display   = 'none';

  try {
    const res = await fetch(`${API}/api/historical-backtest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        symbol, historical_date: date, horizon, asset_type: assetType
      }),
    });
    const data = await res.json();

    document.getElementById('bt-loading').style.display = 'none';

    if (data.status === 'error') {
      const errEl = document.getElementById('bt-error');
      errEl.textContent = data.error || 'Unknown backtest error';
      errEl.style.display = 'block';
      return;
    }

    renderBacktestResults(data);

  } catch (err) {
    document.getElementById('bt-loading').style.display = 'none';
    const errEl = document.getElementById('bt-error');
    errEl.textContent = `Connection failed: ${err.message}`;
    errEl.style.display = 'block';
  }
}

function renderBacktestResults(data) {
  lastBacktestData = data;
  const { prediction, actual, accuracy, chart, symbol, historical_date, horizon_days } = data;

  // Predicted box
  document.getElementById('bt-res-date').textContent = historical_date;

  const predDir    = prediction.predicted_direction || '—';
  const predColor  = predDir === 'SUBE' ? 'var(--positive)' : predDir === 'BAJA' ? 'var(--negative)' : 'var(--text-secondary)';
  const predReturn = prediction.predicted_return_pct ?? 0;

  document.getElementById('bt-pred-direction').textContent = predDir === 'SUBE' ? '▲ SUBE' : predDir === 'BAJA' ? '▼ BAJA' : '→ LATERAL';
  document.getElementById('bt-pred-direction').style.color = predColor;
  document.getElementById('bt-pred-return').textContent    = `${predReturn >= 0 ? '+' : ''}${predReturn.toFixed(2)}% expected`;
  document.getElementById('bt-pred-target').textContent    = `Target: $${fmtNum(prediction.target_price)}`;
  document.getElementById('bt-pred-confidence').textContent= `${Math.round((prediction.confidence || 0) * 100)}% model confidence`;

  // Actual box
  const actDir    = actual.actual_direction || '—';
  const actColor  = actDir === 'SUBE' ? 'var(--positive)' : actDir === 'BAJA' ? 'var(--negative)' : 'var(--text-secondary)';
  const actReturn = actual.actual_return_pct ?? 0;

  document.getElementById('bt-actual-direction').textContent = actDir === 'SUBE' ? '▲ SUBE' : actDir === 'BAJA' ? '▼ BAJA' : '→ LATERAL';
  document.getElementById('bt-actual-direction').style.color = actColor;
  document.getElementById('bt-actual-return').textContent    = `${actReturn >= 0 ? '+' : ''}${actReturn.toFixed(2)}% actual return`;
  document.getElementById('bt-actual-price').textContent     = `Exit price: $${fmtNum(actual.exit_price)}`;

  const accBadge  = document.getElementById('bt-accuracy-badge');
  const correct   = accuracy.direction_correct;
  accBadge.textContent  = correct ? '✓ DIRECTION CORRECT' : '✗ DIRECTION WRONG';
  accBadge.className    = 'bt-acc-badge ' + (correct ? 'acc-correct' : 'acc-incorrect');

  // Metrics row
  document.getElementById('bt-dir-correct').textContent  = correct ? '✓ YES' : '✗ NO';
  document.getElementById('bt-dir-correct').style.color  = correct ? 'var(--positive)' : 'var(--negative)';
  document.getElementById('bt-return-error').textContent = (accuracy.return_error_pct ?? 0).toFixed(2) + '%';
  document.getElementById('bt-rating').textContent       = accuracy.return_error_rating || '—';
  const ratingColor = { EXCELLENT: 'var(--positive)', GOOD: 'var(--warning)', POOR: 'var(--negative)' };
  document.getElementById('bt-rating').style.color = ratingColor[accuracy.return_error_rating] || 'var(--text-primary)';
  document.getElementById('bt-models-used').textContent  = `${prediction.ml_models?.length ?? 0} models`;

  // ── Dual-line chart with Monte Carlo confidence bands ──────────────────
  if (btChart) btChart.destroy();
  const hasCI = Array.isArray(chart.ci_lower) && chart.ci_lower.length > 0;

  const datasets = [];

  // CI band fill (10th–90th percentile range, visually prominent)
  if (hasCI) {
    // Upper envelope — fills DOWN to lower envelope
    datasets.push({
      label: '90th Percentile',
      data: chart.ci_upper,
      borderColor: 'rgba(13,207,207,0.35)',
      borderWidth: 1,
      borderDash: [3, 3],
      pointRadius: 0,
      fill: '+1',                            // fill to next dataset (10th pct)
      backgroundColor: 'rgba(13,207,207,0.12)',
      tension: 0.4,
    });
    // Lower envelope
    datasets.push({
      label: '10th Percentile',
      data: chart.ci_lower,
      borderColor: 'rgba(13,207,207,0.35)',
      borderWidth: 1,
      borderDash: [3, 3],
      pointRadius: 0,
      fill: false,
      tension: 0.4,
    });
  }

  // Predicted median path
  datasets.push({
    label: 'Predicted Path (median)',
    data: chart.predicted_path,
    borderColor: '#0dcfcf',
    borderWidth: 2.5,
    borderDash: [8, 4],
    pointRadius: 0,
    fill: false,
    tension: 0.4,
  });

  // Actual price path
  datasets.push({
    label: 'Actual Price Path',
    data: chart.actual_path,
    borderColor: '#f1f5f9',
    borderWidth: 2.5,
    pointRadius: 0,
    fill: false,
    tension: 0.3,
  });

  btChart = new Chart(document.getElementById('bt-chart').getContext('2d'), {
    type: 'line',
    data: { labels: chart.dates, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: true,
          labels: {
            color: '#94a3b8', font: { size: 11 },
            // Hide the CI envelope lines (keep median + actual)
            filter: item => item.text !== '90th Percentile' && item.text !== '10th Percentile',
          }
        },
        tooltip: {
          backgroundColor: 'rgba(15,20,33,0.95)',
          borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1,
          titleColor: '#f1f5f9', bodyColor: '#94a3b8',
          callbacks: {
            label: ctx => {
              if (ctx.dataset.label.includes('th Percentile')) return null;
              return ` ${ctx.dataset.label}: $${fmtNum(ctx.parsed.y)}`;
            }
          }
        }
      },
      scales: {
        x: { ticks: { color: '#475569', maxTicksLimit: 10 }, grid: { color: 'rgba(255,255,255,0.03)' } },
        y: {
          ticks: { color: '#475569', callback: v => '$' + fmtNum(v) },
          grid: { color: 'rgba(255,255,255,0.04)' }
        }
      }
    }
  });
  document.getElementById('bt-chart').parentElement.style.height = '320px';

  document.getElementById('bt-results').style.display = 'block';
  document.getElementById('bt-results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ══════════════════════════════════════════════
//  CALCULATION EXPLANATION MODAL
// ══════════════════════════════════════════════
function setupCalcModal() {
  const modal   = document.getElementById('calc-modal');
  const close   = document.getElementById('modal-close');
  const backdrop= document.getElementById('modal-backdrop');

  close.addEventListener('click', closeCalcModal);
  backdrop.addEventListener('click', closeCalcModal);
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && modal.style.display !== 'none') closeCalcModal();
  });

  // Main analysis explain button
  document.getElementById('calc-explain-btn').addEventListener('click', () => {
    if (!lastAnalysisData) return;
    openCalcModal(buildAnalysisExplanation(lastAnalysisData));
  });

  // Projected price explain button
  document.getElementById('proj-how-btn').addEventListener('click', () => {
    if (!lastAnalysisData) return;
    openCalcModal(buildAnalysisExplanation(lastAnalysisData));
  });

  // Backtest explain button — delegate since results appear later
  document.addEventListener('click', e => {
    if (e.target && e.target.id === 'bt-explain-btn') {
      if (!lastBacktestData) return;
      openCalcModal(buildBacktestExplanation(lastBacktestData));
    }
  });
}

function openCalcModal(html) {
  document.getElementById('modal-body').innerHTML = html;
  document.getElementById('calc-modal').style.display = 'flex';
  document.body.style.overflow = 'hidden';
}

function closeCalcModal() {
  document.getElementById('calc-modal').style.display = 'none';
  document.body.style.overflow = '';
}

function tag(text, type = '') {
  return `<span class="modal-tag ${type}">${text}</span>`;
}
function row(key, val) {
  return `<div class="modal-row"><span class="modal-key">${key}</span><span class="modal-val">${val}</span></div>`;
}

function buildAnalysisExplanation(data) {
  const info   = data.data_info  || {};
  const ml     = data.ml         || {};
  const hybrid = data.hybrid     || {};
  const agents = data.agents?.individual || {};
  const symbol = data.symbol     || '—';
  const type   = data.asset_type || 'stock';

  const price  = info.current_price;
  const ens    = ml.ensemble_prediction;
  const target = (price && ens !== undefined) ? '$' + fmtNum(price * (1 + ens)) : '—';
  const models = ml.models || [];
  const preds  = ml.predictions || {};

  const recMap = { COMPRA_FUERTE: 'Strong Buy', COMPRA: 'Buy', MANTENER: 'Hold', VENTA: 'Sell', VENTA_FUERTE: 'Strong Sell' };
  const recTag = (r) => {
    const type = r?.includes('COMPRA') ? 'pos' : r?.includes('VENTA') ? 'neg' : 'warn';
    return tag(recMap[r] || r || '—', type);
  };

  let modelRows = '';
  models.forEach(m => {
    const pred = preds[m.name];
    const pctStr = pred !== undefined ? ((pred * 100) >= 0 ? '+' : '') + (pred * 100).toFixed(2) + '%' : '—';
    const dir    = pred > 0.001 ? '↑ Long' : pred < -0.001 ? '↓ Short' : '→ Neutral';
    modelRows += row(`${m.name}`, `${pctStr} ${dir} · RMSE ${m.rmse?.toFixed(3) ?? '—'} · R² ${m.r2?.toFixed(3) ?? '—'}`);
  });
  if (ml.lstm_available) modelRows += row('LSTM Recurrent', 'Integrated into ensemble stack');

  const scoreStr = hybrid.score?.toFixed(3) ?? '—';
  const confPct  = Math.round((hybrid.confidence || 0) * 100);
  const horizonDays = ml.prediction_horizon_days || 5;

  let agentSection = '';
  [
    { key: 'technical',    label: 'Technical Analysis' },
    { key: 'sentiment',    label: 'Sentiment Insights' },
    { key: 'risk',         label: 'Risk Management' },
    { key: 'ml_prediction',label: 'ML Engine' },
  ].forEach(({ key, label }) => {
    const ag = agents[key] || {};
    const conf = ag.confidence ? Math.round(ag.confidence * 100) + '%' : '—';
    agentSection += row(label, `${recTag(ag.recommendation)} · Confidence ${conf}`);
    if (ag.reasoning) {
      agentSection += `<p style="font-size:0.78rem;color:var(--text-muted);margin:0 0 0.5rem 140px;font-style:italic">"${ag.reasoning.slice(0, 180)}${ag.reasoning.length > 180 ? '…' : ''}"</p>`;
    }
  });

  const scoreBound = hybrid.score >= 0.3 ? 'above +0.3 → Buy signal' : hybrid.score <= -0.3 ? 'below -0.3 → Sell signal' : 'between -0.3 and +0.3 → Hold';
  const conflictNote = hybrid.direction_conflict
    ? `<p style="margin-top:0.5rem;color:var(--warning)">⚠ <strong>Direction conflict detected:</strong> the ML model and LLM agents disagreed on direction. The recommendation was automatically capped at MANTENER to avoid a false Buy/Sell signal.</p>`
    : '';

  return `
    <h4>📊 Asset & Data</h4>
    ${row('Asset', `${symbol} (${type === 'crypto' ? 'Cryptocurrency' : 'Equity'})`)}
    ${row('Data points used', `${info.records || '—'} daily closes`)}
    ${row('Analysis period', data.timeframe?.toUpperCase() || '—')}
    ${row('Current price', price ? '$' + fmtNum(price) : '—')}

    <h4>⚙️ Feature Engineering</h4>
    <p><strong>${ml.n_features || '55+'}  technical indicators</strong> were calculated from price and volume data:</p>
    <p>Trend: SMA(20/50/200), EMA(12/26), MACD, ADX &nbsp;·&nbsp; Momentum: RSI, Stochastic %K/%D, Williams %R, Rate of Change &nbsp;·&nbsp; Volume: OBV, Volume SMA ratio &nbsp;·&nbsp; Volatility: Bollinger Bands, ATR &nbsp;·&nbsp; Pattern: Ichimoku Cloud, Momentum(5/10/20)</p>

    <h4>🤖 Machine Learning Models</h4>
    <p>Each model was trained using <strong>TimeSeriesSplit(5) walk-forward validation</strong> — respecting time order so the model never trains on future data.</p>
    ${modelRows}

    <h4>📐 Ensemble Combination</h4>
    ${row('Method', ml.stacking_used ? 'Ridge meta-learner (stacking on out-of-fold predictions)' : 'Weighted average')}
    ${row('Ensemble prediction', ens !== undefined ? ((ens * 100) >= 0 ? '+' : '') + (ens * 100).toFixed(2) + '%' : '—')}
    ${row('Horizon', `${horizonDays} days`)}
    ${row('Projected price', target)}

    <h4>🧠 AI Agent Decisions</h4>
    ${agentSection}

    <h4>⚖️ Final Decision</h4>
    ${row('Hybrid score', `${scoreStr} (range: −1 bearish → +1 bullish)`)}
    ${row('Confidence', `${confPct}%`)}
    ${row('Signal logic', scoreBound)}
    ${row('Recommendation', recTag(hybrid.recommendation))}
    <p style="margin-top:0.5rem">The hybrid score blends the <strong>ML ensemble return prediction</strong> (60% weight, scaled to [-1, 1]) with the <strong>LLM agent consensus</strong> (40% weight). A score above <strong>+0.3</strong> triggers a Buy; below <strong>−0.3</strong> a Sell; otherwise Hold.</p>
    <p style="margin-top:0.35rem"><strong>Direction-consistency rule:</strong> if the ML model and LLM agents predict opposite directions, the recommendation is automatically capped at MANTENER — the system will never say BUY when the price projection is downward.</p>
    ${conflictNote}

    <p class="modal-note">⚠️ This is a research model for educational purposes. Past predictions are not a guarantee of future results. Always apply your own judgement before making financial decisions.</p>
  `;
}

function buildBacktestExplanation(data) {
  const { prediction, actual, accuracy, symbol, historical_date, horizon_days } = data;
  const correct = accuracy?.direction_correct;

  return `
    <h4>🕰️ Backtest Setup</h4>
    ${row('Asset', symbol || '—')}
    ${row('Historical date ("Fake Today")', historical_date || '—')}
    ${row('Prediction horizon', `${horizon_days} days`)}
    <p>The model was given <strong>only data available before ${historical_date}</strong>. It trained on a 2-year window ending exactly on that date — no future data was used at any point.</p>

    <h4>🤖 What the Model Predicted</h4>
    ${row('Direction', prediction?.predicted_direction === 'SUBE' ? '▲ UP' : '▼ DOWN')}
    ${row('Expected return', ((prediction?.predicted_return_pct ?? 0) >= 0 ? '+' : '') + (prediction?.predicted_return_pct ?? 0).toFixed(2) + '%')}
    ${row('Target price', prediction?.target_price ? '$' + fmtNum(prediction.target_price) : '—')}
    ${row('Model confidence', Math.round((prediction?.confidence ?? 0) * 100) + '%')}
    ${row('Models used', (prediction?.ml_models ?? []).join(', ') || '—')}

    <h4>📈 What Actually Happened</h4>
    ${row('Actual direction', actual?.actual_direction === 'SUBE' ? '▲ UP' : '▼ DOWN')}
    ${row('Actual return', ((actual?.actual_return_pct ?? 0) >= 0 ? '+' : '') + (actual?.actual_return_pct ?? 0).toFixed(2) + '%')}
    ${row('Exit price', actual?.exit_price ? '$' + fmtNum(actual.exit_price) : '—')}

    <h4>✅ Accuracy Assessment</h4>
    ${row('Direction correct?', correct ? tag('YES — direction matched', 'pos') : tag('NO — direction wrong', 'neg'))}
    ${row('Return prediction error', (accuracy?.return_error_pct ?? 0).toFixed(2) + '%')}
    ${row('Accuracy rating', tag(accuracy?.return_error_rating || '—', accuracy?.return_error_rating === 'EXCELLENT' ? 'pos' : accuracy?.return_error_rating === 'GOOD' ? 'warn' : 'neg'))}
    <p style="margin-top:0.5rem">The chart shows the model's <span style="color:var(--accent-teal)">predicted price path</span> (a linear projection from entry price to target) against the <span style="color:var(--text-primary)">actual recorded price path</span> for the same period.</p>

    <p class="modal-note">⚠️ Backtesting has inherent limitations. A model can perform well historically and still fail going forward due to changing market regimes. This tool is for educational analysis only.</p>
  `;
}

// ══════════════════════════════════════════════
//  HELPERS
// ══════════════════════════════════════════════
function translateRec(rec) {
  const map = {
    COMPRA_FUERTE: 'COMPRA',
    COMPRA:        'COMPRA',
    MANTENER:      'MANTENER',
    VENTA:         'VENTA',
    VENTA_FUERTE:  'VENTA',
  };
  return map[rec] || rec;
}

function recColor(rec) {
  if (rec.includes('COMPRA')) return 'var(--positive)';
  if (rec.includes('VENTA'))  return 'var(--negative)';
  return 'var(--warning)';
}

function fmtNum(n) {
  if (n === null || n === undefined) return '—';
  if (n < 0.01) return n.toFixed(6);
  if (n < 1)    return n.toFixed(4);
  return n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function showLoading(visible) {
  document.getElementById('loading-state').style.display = visible ? 'block' : 'none';
  if (!visible) {
    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('loading-text').textContent = 'Analyzing market data...';
    document.getElementById('loading-sub').textContent  = 'Initializing pipeline';
  }
}
function hideResults()  { document.getElementById('analysis-section').style.display = 'none'; }
function hideError()    { document.getElementById('error-state').style.display = 'none'; }
function showError(msg) {
  document.getElementById('error-message').textContent = msg;
  document.getElementById('error-state').style.display = 'block';
  document.getElementById('error-state').scrollIntoView({ behavior: 'smooth' });
}

// ══════════════════════════════════════════════
//  TRADING BOT — Paper Trading Dashboard
// ══════════════════════════════════════════════

let _botSignalData  = null;   // última señal obtenida
let _botRefreshTmr  = null;   // temporizador de refresco automático
let _hbtChart       = null;   // Chart.js del backtest histórico
let _hbtDdChart     = null;   // Chart.js drawdown del backtest histórico
let _scanExtraSyms  = [];     // símbolos extra del scanner
let _scanType       = 'both'; // tipo de activo del scanner

// ── Inicialización ────────────────────────────
function setupBotView() {
  // ── Tabs ────────────────────────────────────
  document.querySelectorAll('.bot-tab').forEach(tab => {
    tab.addEventListener('click', () => botSwitchTab(tab.dataset.tab));
  });

  // ── Live paper trading ───────────────────────
  document.getElementById('bot-auto-btn')
    .addEventListener('click', botAutoScanAndExecute);
  document.getElementById('bot-reset-btn')
    .addEventListener('click', botResetPortfolio);
  document.getElementById('bot-refresh-btn')
    .addEventListener('click', () => botLoadStatus(true));

  // Collapsible manual section
  document.getElementById('bot-manual-toggle').addEventListener('click', () => {
    const sec = document.getElementById('bot-manual-section');
    const tog = document.getElementById('bot-manual-toggle');
    const open = sec.style.display !== 'none';
    sec.style.display = open ? 'none' : 'block';
    tog.textContent   = open
      ? '▸ o analiza un activo específico manualmente'
      : '▾ o analiza un activo específico manualmente';
  });

  document.getElementById('bot-analyze-btn')
    .addEventListener('click', botGetSignal);
  document.getElementById('bot-symbol-input')
    .addEventListener('keydown', e => { if (e.key === 'Enter') botGetSignal(); });

  const slider = document.getElementById('bsc-size-slider');
  slider.addEventListener('input', () => {
    document.getElementById('bsc-size-display').textContent = slider.value + '%';
    if (_botSignalData && _botSignalData.action !== 'HOLD') {
      document.getElementById('bsc-execute-btn').textContent =
        `Ejecutar ${_botSignalData.action} — ${slider.value}% del capital`;
    }
  });
  document.getElementById('bsc-execute-btn').addEventListener('click', botExecuteTrade);

  document.querySelectorAll('.bot-pick-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.getElementById('bot-symbol-input').value = btn.dataset.symbol;
      document.getElementById('bot-asset-type').value   = btn.dataset.type;
      botGetSignal();
    });
  });

  // ── Historical backtest ──────────────────────
  // Default dates: 1 year ago → today
  const now  = new Date();
  const yago = new Date(now); yago.setFullYear(now.getFullYear() - 1);
  document.getElementById('hbt-end-date').value   = now.toISOString().slice(0,10);
  document.getElementById('hbt-start-date').value = yago.toISOString().slice(0,10);

  document.getElementById('hbt-run-btn').addEventListener('click', hbtRun);
  document.getElementById('hbt-symbol')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') hbtRun();
  });

  // Period quick picks
  document.querySelectorAll('.hbt-period-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const months = parseInt(btn.dataset.months);
      const end    = new Date();
      const start  = new Date(end);
      start.setMonth(start.getMonth() - months);
      document.getElementById('hbt-end-date').value   = end.toISOString().slice(0,10);
      document.getElementById('hbt-start-date').value = start.toISOString().slice(0,10);
    });
  });

  // Symbol quick picks for HBT
  document.querySelectorAll('.hbt-sym-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.getElementById('hbt-symbol').value         = btn.dataset.symbol;
      document.getElementById('hbt-asset-type').value     = btn.dataset.type;
    });
  });

  // ── Scanner ──────────────────────────────────
  document.getElementById('scan-run-btn').addEventListener('click', scanRun);

  document.querySelectorAll('#scan-type-toggle .asset-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('#scan-type-toggle .asset-pill')
        .forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      _scanType = pill.dataset.scanType;
    });
  });

  document.getElementById('scan-add-symbol-btn').addEventListener('click', () => {
    const sym  = document.getElementById('scan-extra-symbol').value.trim().toUpperCase();
    const type = document.getElementById('scan-extra-type').value;
    if (sym && !_scanExtraSyms.find(s => s.symbol === sym)) {
      _scanExtraSyms.push({ symbol: sym, asset_type: type });
      document.getElementById('scan-extra-symbol').value = '';
      botShowToast(`${sym} añadido al scanner`, 'success');
    }
  });

  // ── Session management ───────────────────────
  document.getElementById('bot-session-save-btn')
    ?.addEventListener('click', botSaveSession);
  document.getElementById('bot-session-load-btn')
    ?.addEventListener('click', botLoadSession);
  document.getElementById('bot-session-delete-btn')
    ?.addEventListener('click', botDeleteSession);
  botLoadSessions();
}

// ══════════════════════════════════════════════
//  SESSION MANAGEMENT
// ══════════════════════════════════════════════

let _liveAnalysisInterval = null;

async function botLoadSessions() {
  try {
    const res  = await fetch(`${API}/api/paper/sessions`);
    const data = await res.json();
    if (data.status !== 'success') return;
    const sel = document.getElementById('bot-session-select');
    if (!sel) return;
    // Keep first default option
    sel.innerHTML = '<option value="">Sesión activa (sin guardar)</option>';
    (data.sessions || []).forEach(s => {
      const opt = document.createElement('option');
      opt.value       = s.session_name;
      const saved     = s.saved_at ? new Date(s.saved_at).toLocaleDateString('es-ES') : '';
      opt.textContent = `${s.session_name} · ${s.closed_trades} trades · ${saved}`;
      sel.appendChild(opt);
    });
  } catch (e) { console.warn('[Sessions] Error cargando sesiones:', e); }
}

async function botSaveSession() {
  const name = (document.getElementById('bot-session-name')?.value || '').trim();
  if (!name) { botShowToast('Introduce un nombre para la sesión', 'error'); return; }
  try {
    const res  = await fetch(`${API}/api/paper/sessions/save`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    const data = await res.json();
    if (data.status !== 'success') throw new Error(data.error);
    botShowToast(`Sesión "${name}" guardada`, 'success');
    document.getElementById('bot-session-name').value = '';
    await botLoadSessions();
  } catch (e) { botShowToast('Error al guardar: ' + e.message, 'error'); }
}

async function botLoadSession() {
  const name = document.getElementById('bot-session-select')?.value;
  if (!name) { botShowToast('Selecciona una sesión para cargar', 'error'); return; }
  if (!confirm(`¿Cargar la sesión "${name}"? Se sobreescribirá el estado actual.`)) return;
  try {
    const res  = await fetch(`${API}/api/paper/sessions/load`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    const data = await res.json();
    if (data.status !== 'success') throw new Error(data.error);
    botRenderOverview(data.portfolio);
    botRenderPositions(data.portfolio.positions_with_pnl || []);
    botRenderHistory(data.portfolio);
    botShowToast(`Sesión "${name}" cargada`, 'success');
  } catch (e) { botShowToast('Error al cargar: ' + e.message, 'error'); }
}

async function botDeleteSession() {
  const name = document.getElementById('bot-session-select')?.value;
  if (!name) { botShowToast('Selecciona una sesión para eliminar', 'error'); return; }
  if (!confirm(`¿Eliminar la sesión "${name}"? Esta acción no se puede deshacer.`)) return;
  try {
    const res  = await fetch(`${API}/api/paper/sessions/delete`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    const data = await res.json();
    if (data.status !== 'success') throw new Error(data.error);
    botShowToast(`Sesión "${name}" eliminada`, 'success');
    await botLoadSessions();
  } catch (e) { botShowToast('Error al eliminar: ' + e.message, 'error'); }
}

// ══════════════════════════════════════════════
//  LIVE ANALYSIS (auto-refresh every 60s)
// ══════════════════════════════════════════════

async function botRunLiveAnalysis() {
  const card = document.getElementById('bot-live-analysis-card');
  const grid = document.getElementById('bot-live-analysis-grid');
  if (!card || !grid) return;

  try {
    const res  = await fetch(`${API}/api/paper/live-analysis`);
    const data = await res.json();
    if (data.status !== 'success') return;

    const positions = data.portfolio?.positions_with_pnl || [];
    if (positions.length === 0) { card.style.display = 'none'; return; }

    card.style.display = 'block';
    const analysis = data.analysis || [];

    grid.innerHTML = analysis.map(a => {
      if (a.error) return `<div class="bot-live-item error"><strong>${a.symbol}</strong>: ${a.error}</div>`;
      const pnlColor = a.unrealized_pnl >= 0 ? 'var(--positive)' : 'var(--negative)';
      const dirColor = a.ml_direction === 'ALCISTA' ? 'var(--positive)' : 'var(--negative)';
      const closeWarn = a.should_close
        ? `<div class="bot-live-close-warn">⚠️ El modelo sugiere CERRAR esta posición (señal invertida)</div>` : '';
      return `<div class="bot-live-item ${a.should_close ? 'warn' : ''}">
        <div class="bot-live-header">
          <strong>${a.symbol}</strong>
          <span class="bot-live-pnl" style="color:${pnlColor}">
            ${a.unrealized_pnl >= 0 ? '+' : ''}$${fmtNum(Math.abs(a.unrealized_pnl))}
            (${a.unrealized_pnl_pct >= 0 ? '+' : ''}${a.unrealized_pnl_pct.toFixed(2)}%)
          </span>
        </div>
        <div class="bot-live-row">
          <span class="bot-live-label">ML hoy:</span>
          <span style="color:${dirColor}">${a.ml_direction} ${a.ml_pred_pct >= 0 ? '+' : ''}${a.ml_pred_pct.toFixed(2)}%</span>
          <span class="bot-live-conf">(conf. ${(a.ml_confidence*100).toFixed(0)}%)</span>
        </div>
        <div class="bot-live-row">
          <span class="bot-live-label">Técnico:</span>
          <span>${a.tech_rec || '—'}</span>
        </div>
        <div class="bot-live-row">
          <span class="bot-live-label">Riesgo:</span>
          <span>${a.risk_rec || '—'}</span>
        </div>
        ${a.tech_reasoning ? `<div class="bot-live-reasoning">${a.tech_reasoning.slice(0,150)}</div>` : ''}
        ${closeWarn}
      </div>`;
    }).join('');

    // Update portfolio overview with live prices
    botRenderOverview(data.portfolio);
    botRenderPositions(data.portfolio.positions_with_pnl || []);
  } catch (e) { console.warn('[LiveAnalysis] Error:', e); }
}

function botStartLiveAnalysis() {
  botRunLiveAnalysis();
  if (_liveAnalysisInterval) clearInterval(_liveAnalysisInterval);
  _liveAnalysisInterval = setInterval(botRunLiveAnalysis, 60000);  // every 60s
}

// ── Tab switching ─────────────────────────────
function botSwitchTab(tab) {
  document.querySelectorAll('.bot-tab').forEach(t =>
    t.classList.toggle('active', t.dataset.tab === tab));
  document.querySelectorAll('.bot-tab-panel').forEach(p =>
    p.style.display = 'none');
  const panel = document.getElementById('bot-tab-' + tab);
  if (panel) panel.style.display = 'block';
}

// ── Cargar estado del portfolio ───────────────
async function botLoadStatus(live = false) {
  try {
    const res  = await fetch(`${API}/api/paper/status?live=${live}`);
    const data = await res.json();
    if (data.status !== 'success') throw new Error(data.error || 'Error al cargar portfolio');
    botRenderOverview(data);
    botRenderPositions(data.positions_with_pnl || []);
    botRenderHistory(data);
    // Start live analysis polling when positions exist
    if ((data.positions_with_pnl || []).length > 0) {
      botStartLiveAnalysis();
    }
  } catch (err) {
    console.warn('[Bot] Error cargando estado:', err);
  }
}

// ── Renderizar banner de overview ─────────────
function botRenderOverview(d) {
  const pnlTotal = (d.unrealized_pnl || 0) + (d.realized_pnl || 0);
  const ret      = d.total_return_pct || 0;
  const retSign  = ret >= 0;
  const pnlSign  = pnlTotal >= 0;

  setText('bot-equity',         '$' + fmtNum(d.total_equity));
  setText('bot-cash',           '$' + fmtNum(d.cash));
  setColorVal('bot-unrealized', d.unrealized_pnl,  true);
  setColorVal('bot-realized',   d.realized_pnl,    true);

  const retEl = document.getElementById('bot-return');
  retEl.textContent = (retSign ? '+' : '') + ret.toFixed(2) + '%';
  retEl.style.color = retSign ? 'var(--positive)' : 'var(--negative)';

  setText('bot-open',         d.num_open);
  setText('bot-trades-count', d.num_trades);
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function setColorVal(id, val, isCurrency = false) {
  const el = document.getElementById(id);
  if (!el) return;
  const sign = val >= 0;
  el.textContent = (sign ? '+' : '-') + (isCurrency ? '$' : '') + fmtNum(Math.abs(val));
  el.style.color = sign ? 'var(--positive)' : 'var(--negative)';
}

// ── Renderizar posiciones abiertas ────────────
function botRenderPositions(positions) {
  const list  = document.getElementById('bot-positions-list');
  const empty = document.getElementById('bot-positions-empty');

  if (!positions || positions.length === 0) {
    empty.style.display = 'block';
    list.innerHTML      = '';
    return;
  }

  empty.style.display = 'none';
  list.innerHTML = positions.map(pos => {
    const pnl     = pos.unrealized_pnl;
    const pnlPct  = pos.unrealized_pnl_pct;
    const color   = pnl >= 0 ? 'var(--positive)' : 'var(--negative)';
    const entryDate = botFmtDate(pos.entry_date);
    return `
      <div class="bot-position-card">
        <div class="bpc-left">
          <span class="bpc-symbol">${pos.symbol}</span>
          <span class="bpc-action ${pos.action.toLowerCase()}">${pos.action}</span>
          <span class="bpc-shares">${parseFloat(pos.shares).toFixed(4)} unidades</span>
          <span class="bpc-entry-date">Desde ${entryDate}</span>
        </div>
        <div class="bpc-center">
          <div class="bpc-row">Entrada: <strong>$${fmtNum(pos.entry_price)}</strong></div>
          <div class="bpc-row">Ahora:   <strong>$${fmtNum(pos.current_price)}</strong></div>
        </div>
        <div class="bpc-right">
          <div class="bpc-pnl" style="color:${color}">
            ${pnl >= 0 ? '+' : '-'}$${fmtNum(Math.abs(pnl))}
          </div>
          <div class="bpc-pnl-pct" style="color:${color}">
            ${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%
          </div>
        </div>
        <button class="close-pos-btn" onclick="botClosePosition('${pos.symbol}')">Cerrar</button>
      </div>`;
  }).join('');
}

// ── Auto: escanear todos los activos y ejecutar el mejor ─────────────────
async function botAutoScanAndExecute() {
  const btn = document.getElementById('bot-auto-btn');
  btn.querySelector('.btn-text').style.display  = 'none';
  btn.querySelector('.btn-loader').style.display = 'inline-block';
  btn.disabled = true;
  document.getElementById('bot-signal-loading').style.display = 'flex';
  document.getElementById('bot-signal-error').style.display   = 'none';
  document.getElementById('bot-auto-result').style.display    = 'none';

  try {
    // 1) Escanear todos los activos del watchlist
    const scanRes  = await fetch(`${API}/api/paper/scan`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ asset_type: 'both', extra_symbols: [] }),
    });
    const scanData = await scanRes.json();
    if (scanData.status !== 'success') throw new Error(scanData.error || 'Error en escaneo');

    // 2) Filtrar señales con acción real (no HOLD) y ordenar por fuerza
    const signals = (scanData.signals || []).filter(s => s.action !== 'HOLD');
    if (signals.length === 0) {
      botRenderAutoResult({ type: 'no_signal', scanned: scanData.scanned || 0 });
      return;
    }
    const best = signals[0]; // ya llegan ordenadas por signal_strength desc

    // 3) Ejecutar el mejor trade automáticamente con el tamaño sugerido por Kelly
    const execRes  = await fetch(`${API}/api/paper/execute`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        symbol:     best.symbol,
        asset_type: best.asset_type,
        action:     best.action,
        size_pct:   best.suggested_size_pct || 10,
      }),
    });
    const execData = await execRes.json();
    if (execData.status !== 'success') throw new Error(execData.error || 'Error al ejecutar');

    botRenderAutoResult({ type: 'executed', best, exec: execData, scanned: scanData.scanned || 0 });
    botShowToast(`Trade ejecutado: ${best.action} ${best.symbol}`, 'success');
    botLoadStatus(false);

  } catch (err) {
    document.getElementById('bot-signal-error').style.display  = 'block';
    document.getElementById('bot-signal-error').textContent    = err.message;
  } finally {
    btn.querySelector('.btn-text').style.display  = '';
    btn.querySelector('.btn-loader').style.display = 'none';
    btn.disabled = false;
    document.getElementById('bot-signal-loading').style.display = 'none';
  }
}

function botRenderAutoResult({ type, best, exec, scanned }) {
  const el = document.getElementById('bot-auto-result');
  el.style.display = 'block';

  if (type === 'no_signal') {
    el.innerHTML = `
      <div class="bot-ar-empty">
        <div class="bot-ar-icon">🔍</div>
        <div class="bot-ar-msg">Se escanearon <strong>${scanned}</strong> activos — ninguna señal supera el umbral de confianza ahora mismo.</div>
        <div class="bot-ar-sub">El bot espera a señales de alta convicción para no sobreoperar.</div>
      </div>`;
    return;
  }

  const prSign = best.predicted_return_pct >= 0 ? '+' : '';
  const prCol  = best.predicted_return_pct >= 0 ? 'var(--positive)' : 'var(--negative)';
  const actionCls = best.action === 'LONG' ? 'long' : 'short';

  el.innerHTML = `
    <div class="bot-ar-header">
      <span class="bsc-action ${actionCls}">${best.action}</span>
      <span class="bot-ar-symbol">${best.symbol}</span>
      <span class="bot-ar-rank">Mejor de ${scanned} activos escaneados</span>
    </div>
    <div class="bsc-metrics">
      <div class="bsc-metric">
        <div class="bsc-mlabel">RETORNO PRED.</div>
        <div class="bsc-mval" style="color:${prCol}">${prSign}${best.predicted_return_pct.toFixed(2)}%</div>
      </div>
      <div class="bsc-metric">
        <div class="bsc-mlabel">CONFIANZA</div>
        <div class="bsc-mval">${(best.confidence * 100).toFixed(0)}%</div>
      </div>
      <div class="bsc-metric">
        <div class="bsc-mlabel">FUERZA SEÑAL</div>
        <div class="bsc-mval">${best.signal_strength.toFixed(4)}</div>
      </div>
      <div class="bsc-metric">
        <div class="bsc-mlabel">TAMAÑO USADO</div>
        <div class="bsc-mval">${(best.suggested_size_pct || 10).toFixed(1)}%</div>
      </div>
    </div>
    <div class="bsc-reason">${best.reason || ''}</div>
    <div class="bot-ar-exec-ok">✓ Trade ejecutado a $${fmtNum((exec.trade && exec.trade.entry_price) || best.current_price)}</div>`;
}

// ── Señal manual de un activo específico ──────
async function botGetSignal() {
  const symbol    = (document.getElementById('bot-symbol-input').value || '').trim().toUpperCase();
  const assetType = document.getElementById('bot-asset-type').value;
  if (!symbol) { botShowToast('Introduce un símbolo primero', 'error'); return; }

  // UI loading
  const btn = document.getElementById('bot-analyze-btn');
  btn.querySelector('.btn-text').style.display = 'none';
  btn.querySelector('.btn-loader').style.display = 'inline-block';
  btn.disabled = true;
  document.getElementById('bot-signal-card').style.display    = 'none';
  document.getElementById('bot-signal-loading').style.display = 'flex';
  document.getElementById('bot-signal-error').style.display   = 'none';

  try {
    const res  = await fetch(`${API}/api/paper/signal`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ symbol, asset_type: assetType }),
    });
    const data = await res.json();
    if (data.status !== 'success') throw new Error(data.error || 'Error en análisis');

    _botSignalData = data;
    botRenderSignalCard(data);
  } catch (err) {
    document.getElementById('bot-signal-error').style.display = 'block';
    document.getElementById('bot-signal-error').textContent   = err.message;
  } finally {
    btn.querySelector('.btn-text').style.display   = '';
    btn.querySelector('.btn-loader').style.display = 'none';
    btn.disabled = false;
    document.getElementById('bot-signal-loading').style.display = 'none';
  }
}

// ── Renderizar tarjeta de señal ───────────────
function botRenderSignalCard(data) {
  const card = document.getElementById('bot-signal-card');
  card.style.display = 'block';

  // Acción (LONG / SHORT / HOLD)
  const actionEl = document.getElementById('bsc-action');
  actionEl.textContent  = data.action;
  actionEl.className    = 'bsc-action ' + data.action.toLowerCase();

  document.getElementById('bsc-symbol-display').textContent =
    data.symbol + ' · ' + (data.asset_type === 'crypto' ? 'Crypto' : 'Stock');
  document.getElementById('bsc-timestamp').textContent =
    new Date(data.timestamp).toLocaleTimeString('es-ES', { hour:'2-digit', minute:'2-digit' });

  // Métricas
  const pr = data.predicted_return_pct;
  const prEl = document.getElementById('bsc-pred-return');
  prEl.textContent = (pr >= 0 ? '+' : '') + pr.toFixed(2) + '%';
  prEl.style.color = pr >= 0 ? 'var(--positive)' : 'var(--negative)';

  document.getElementById('bsc-conf').textContent  = (data.confidence * 100).toFixed(0) + '%';
  document.getElementById('bsc-ss').textContent    = data.signal_strength.toFixed(4);
  document.getElementById('bsc-price').textContent = data.current_price
    ? '$' + fmtNum(data.current_price) : '—';
  document.getElementById('bsc-reason').textContent = data.reason || '';

  // ── Multi-agent analysis cards ─────────────────
  const agentsSection = document.getElementById('bsc-agents-section');
  const agentsGrid    = document.getElementById('bsc-agents-grid');
  if (data.agents && Object.keys(data.agents).length > 0) {
    const agentLabels = {
      technical:         { icon: '📊', name: 'Analista Técnico' },
      risk:              { icon: '⚠️', name: 'Control de Riesgo' },
      sentiment:         { icon: '📰', name: 'Analista Sentimiento' },
      ml:                { icon: '🤖', name: 'Modelo ML Ensemble' },
      portfolio_manager: { icon: '👔', name: 'Director de Inversiones (LLM)' },
    };
    const recColor = r => {
      const ru = (r || '').toUpperCase();
      if (ru.includes('COMPRA') || ru.includes('BUY') || ru.includes('ALCISTA') || ru.includes('APROBAR'))
        return 'var(--positive)';
      if (ru.includes('VENTA') || ru.includes('SELL') || ru.includes('BAJISTA') || ru.includes('RECHAZ'))
        return 'var(--negative)';
      return 'var(--warning)';
    };
    agentsGrid.innerHTML = Object.entries(data.agents).map(([key, ag]) => {
      const lbl = agentLabels[key] || { icon: '🔍', name: key };
      const conf = ag.confidence ? (ag.confidence * 100).toFixed(0) + '%' : '';
      return `<div class="bsc-agent-card">
        <div class="bsc-agent-header">
          <span class="bsc-agent-icon">${lbl.icon}</span>
          <span class="bsc-agent-name">${lbl.name}</span>
        </div>
        <div class="bsc-agent-rec" style="color:${recColor(ag.recommendation)}">${ag.recommendation || '—'}</div>
        ${conf ? `<div class="bsc-agent-conf">Confianza: ${conf}</div>` : ''}
        ${ag.reasoning ? `<div class="bsc-agent-reasoning">${ag.reasoning.slice(0, 200)}</div>` : ''}
      </div>`;
    }).join('');
    agentsSection.style.display = 'block';
  } else {
    agentsSection.style.display = 'none';
  }

  // Bloque de ejecución
  const execWrap = document.getElementById('bsc-execute-wrap');
  if (data.action !== 'HOLD') {
    execWrap.style.display = 'block';
    const suggested = Math.min(Math.max(Math.round(data.suggested_size_pct), 1), 30);
    document.getElementById('bsc-size-slider').value       = suggested;
    document.getElementById('bsc-size-display').textContent = suggested + '%';
    document.getElementById('bsc-kelly-hint').textContent  = suggested + '% (Kelly)';
    document.getElementById('bsc-execute-btn').textContent =
      `Ejecutar ${data.action} — ${suggested}% del capital`;
  } else {
    execWrap.style.display = 'none';
  }
}

// ── Ejecutar trade ────────────────────────────
async function botExecuteTrade() {
  if (!_botSignalData || _botSignalData.action === 'HOLD') return;

  const sizePct = parseFloat(document.getElementById('bsc-size-slider').value);
  const btn     = document.getElementById('bsc-execute-btn');
  btn.disabled  = true;
  const orig    = btn.textContent;
  btn.textContent = 'Ejecutando...';

  try {
    const res  = await fetch(`${API}/api/paper/execute`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        symbol:               _botSignalData.symbol,
        asset_type:           _botSignalData.asset_type || 'stock',
        action:               _botSignalData.action,
        size_pct:             sizePct,
        signal_strength:      _botSignalData.signal_strength,
        predicted_return_pct: _botSignalData.predicted_return_pct,
        reason:               'SIGNAL',
      }),
    });
    const data = await res.json();
    if (data.status !== 'success') throw new Error(data.error || 'Error al ejecutar');

    // Actualizar UI
    botRenderOverview(data.portfolio);
    botRenderPositions(data.portfolio.positions_with_pnl || []);
    document.getElementById('bot-signal-card').style.display = 'none';
    _botSignalData = null;

    const t = data.trade;
    botShowToast(
      `${t.action} ${t.symbol} ejecutado @ $${fmtNum(t.entry_price)} · ${(sizePct).toFixed(0)}% capital`,
      'success'
    );
  } catch (err) {
    botShowToast('Error: ' + err.message, 'error');
    btn.textContent = orig;
    btn.disabled    = false;
  } finally {
    btn.disabled = false;
  }
}

// ── Cerrar posición ───────────────────────────
async function botClosePosition(symbol) {
  if (!confirm(`¿Cerrar posición en ${symbol} al precio actual de mercado?`)) return;

  try {
    const res  = await fetch(`${API}/api/paper/close`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ symbol }),
    });
    const data = await res.json();
    if (data.status !== 'success') throw new Error(data.error || 'Error al cerrar');

    botRenderOverview(data.portfolio);
    botRenderPositions(data.portfolio.positions_with_pnl || []);
    botRenderHistory(data.portfolio);

    const t   = data.trade;
    const pnl = t.pnl;
    botShowToast(
      `${symbol} cerrado · P&L: ${pnl >= 0 ? '+' : ''}$${fmtNum(Math.abs(pnl))} (${t.return_pct >= 0 ? '+' : ''}${t.return_pct.toFixed(2)}%)`,
      pnl >= 0 ? 'success' : 'error'
    );
  } catch (err) {
    botShowToast('Error: ' + err.message, 'error');
  }
}

// ── Reset portfolio ───────────────────────────
async function botResetPortfolio() {
  if (!confirm('¿Resetear cartera? Se cerrarán todas las posiciones y se volverá a $100,000.')) return;

  try {
    const res  = await fetch(`${API}/api/paper/reset`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ initial_capital: 100000 }),
    });
    const data = await res.json();
    if (data.status !== 'success') throw new Error(data.error || 'Error al resetear');

    botRenderOverview(data.portfolio);
    botRenderPositions(data.portfolio.positions_with_pnl || []);
    botRenderHistory(data.portfolio);
    document.getElementById('bot-signal-card').style.display = 'none';
    _botSignalData = null;
    botShowToast('Cartera reseteada a $100,000', 'success');
  } catch (err) {
    botShowToast('Error: ' + err.message, 'error');
  }
}

// ── Renderizar historial de trades ────────────
function botRenderHistory(portfolio) {
  const trades    = (portfolio.trade_history || []);
  const tbody     = document.getElementById('bot-history-tbody');
  const empty     = document.getElementById('bot-history-empty');
  const tableWrap = document.getElementById('bot-history-table-wrap');

  if (trades.length === 0) {
    empty.style.display     = 'block';
    tableWrap.style.display = 'none';
    return;
  }
  empty.style.display     = 'none';
  tableWrap.style.display = 'block';

  tbody.innerHTML = [...trades].reverse().map(t => {
    const pnl   = t.pnl;
    const color = pnl >= 0 ? 'var(--positive)' : 'var(--negative)';
    return `
      <tr>
        <td><strong>${t.symbol}</strong></td>
        <td><span class="action-badge ${(t.action||'').toLowerCase()}">${t.action}</span></td>
        <td>${botFmtDate(t.entry_date)}</td>
        <td>${botFmtDate(t.exit_date)}</td>
        <td>$${fmtNum(t.entry_price)}</td>
        <td>$${fmtNum(t.exit_price)}</td>
        <td>${t.hold_days}d</td>
        <td style="color:${color};font-weight:600">
          ${pnl >= 0 ? '+' : '-'}$${fmtNum(Math.abs(pnl))}
        </td>
        <td style="color:${color}">
          ${(t.return_pct >= 0 ? '+' : '') + t.return_pct.toFixed(2)}%
        </td>
        <td><span class="reason-badge">${t.exit_reason || '—'}</span></td>
      </tr>`;
  }).join('');
}

// ── Toast ─────────────────────────────────────
function botShowToast(msg, type = 'info') {
  const el = document.getElementById('bot-toast');
  if (!el) return;
  el.textContent = msg;
  el.className   = 'bot-toast visible ' + type;
  setTimeout(() => el.classList.remove('visible'), 5000);
}

// ── Helpers ───────────────────────────────────
function botFmtDate(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleDateString('es-ES', {
      day: '2-digit', month: 'short', year: 'numeric',
    });
  } catch { return iso; }
}

// ══════════════════════════════════════════════
//  HISTORICAL BOT BACKTEST
// ══════════════════════════════════════════════

async function hbtRun() {
  const startDate      = document.getElementById('hbt-start-date').value;
  const endDate        = document.getElementById('hbt-end-date').value;
  const capital        = parseFloat(document.getElementById('hbt-capital').value) || 100000;
  const allowShort     = document.getElementById('hbt-allow-short').value === 'true';
  const commissionRate = parseFloat(document.getElementById('hbt-commission')?.value ?? '0.001');

  if (!startDate || !endDate) { botShowToast('Introduce fechas de inicio y fin', 'error'); return; }
  if (startDate >= endDate)   { botShowToast('La fecha de inicio debe ser anterior al fin', 'error'); return; }

  document.getElementById('hbt-loading').style.display  = 'flex';
  document.getElementById('hbt-results').style.display  = 'none';
  document.getElementById('hbt-error').style.display    = 'none';
  document.getElementById('hbt-run-btn').disabled       = true;

  try {
    // Llama al endpoint autónomo: el bot escanea todos los activos de la watchlist
    const res  = await fetch(`${API}/api/paper/autonomous-backtest`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        start_date: startDate, end_date: endDate,
        initial_capital: capital, allow_short: allowShort,
        // 0.82 (top 18% señales) + Kelly fraccionado 0.22:
        // antes 0.70/0.30 → demasiadas entradas con conviction baja → -22%.
        // Literatura (Faber/Antonacci/Kelly fraccional 0.25×): menos trades,
        // mejor seleccionados → mayor profit factor y menor drawdown.
        signal_percentile: 0.82, kelly_scale: 0.22,
        commission_rate: commissionRate,
        // 'mode' no enviado → backend usa default = 'trend' (ganador del
        // benchmark multi-ventana: avg +23% vs ML −2.66%, 100% ventanas
        // positivas vs 0%, Sharpe 0.51 vs 0.01).
      }),
    });
    const data = await res.json();
    if (data.status !== 'success') throw new Error(data.error || 'Error en backtest');

    hbtRenderResults(data);
    document.getElementById('hbt-results').style.display = 'block';
    document.getElementById('hbt-results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    document.getElementById('hbt-error').style.display   = 'block';
    document.getElementById('hbt-error').textContent     = err.message;
  } finally {
    document.getElementById('hbt-loading').style.display = 'none';
    document.getElementById('hbt-run-btn').disabled      = false;
  }
}

function hbtRenderResults(data) {
  const perf = data.performance || {};
  const ec   = data.equity_curve || {};
  const sig  = data.signal_stats || {};
  const period = data.period || {};

  // ── Period banner ──────────────────────────────────────────────────────
  const banner = document.getElementById('hbt-period-banner');
  const wlStr  = (data.watchlist || []).join(' · ') || 'múltiples activos';
  banner.innerHTML = `
    <div class="hbt-banner-inner">
      <span class="hbt-banner-item">
        <span class="hbt-banner-lbl">ENTRENAMIENTO (cada activo)</span>
        <span class="hbt-banner-val">hasta ${period.train_end || startDate || '?'} · datos anteriores al test</span>
      </span>
      <span class="hbt-banner-arrow">→ TEST →</span>
      <span class="hbt-banner-item">
        <span class="hbt-banner-lbl">PERÍODO DE TEST (out-of-sample)</span>
        <span class="hbt-banner-val">${period.test_start||'?'} → ${period.test_end||'?'} · ${(period.n_test_days||0)} días</span>
      </span>
      <span class="hbt-banner-badge">SIN LEAKAGE</span>
    </div>
    <div style="font-size:0.72rem;color:var(--text-muted);margin-top:0.4rem">
      Activos escaneados: <span style="color:var(--accent-teal)">${wlStr}</span>
    </div>`;

  // ── Metrics grid ───────────────────────────────────────────────────────
  const retPct = perf.total_return_pct || 0;
  const bhFinal = (ec.buy_hold_value || [])[ec.buy_hold_value.length - 1] || 0;
  const bhRet   = bhFinal && perf.initial_capital
    ? ((bhFinal - perf.initial_capital) / perf.initial_capital * 100) : 0;

  const metrics = [
    { l: 'RETORNO TOTAL',       v: (retPct >= 0 ? '+' : '') + retPct.toFixed(2) + '%',
      c: retPct >= 0 ? 'var(--positive)' : 'var(--negative)' },
    { l: 'VS BUY & HOLD',
      v: (retPct - bhRet >= 0 ? '+' : '') + (retPct - bhRet).toFixed(2) + '%',
      c: retPct >= bhRet ? 'var(--positive)' : 'var(--negative)' },
    { l: 'SHARPE RATIO',        v: (perf.sharpe_ratio || 0).toFixed(3),
      c: perf.sharpe_ratio >= 1 ? 'var(--positive)' : perf.sharpe_ratio >= 0 ? 'var(--warning)' : 'var(--negative)' },
    { l: 'SORTINO RATIO',       v: (perf.sortino_ratio || 0).toFixed(3) },
    { l: 'MAX DRAWDOWN',        v: (perf.max_drawdown_pct || 0).toFixed(1) + '%',
      c: 'var(--negative)' },
    { l: 'WIN RATE',            v: (perf.win_rate || 0).toFixed(1) + '%' },
    { l: 'PROFIT FACTOR',       v: (perf.profit_factor || 0).toFixed(3),
      c: perf.profit_factor >= 1.5 ? 'var(--positive)' : perf.profit_factor >= 1 ? 'var(--warning)' : 'var(--negative)' },
    { l: 'CALMAR RATIO',        v: (perf.calmar_ratio || 0).toFixed(3) },
    { l: 'TOTAL TRADES',        v: perf.total_trades || 0 },
    { l: 'AVG HOLD DAYS',       v: (perf.avg_hold_days || 0).toFixed(1) + 'd' },
    { l: 'CAPITAL INICIAL',     v: '$' + fmtNum(perf.initial_capital) },
    { l: 'CAPITAL FINAL',       v: '$' + fmtNum(perf.final_capital),
      c: (perf.final_capital >= perf.initial_capital) ? 'var(--positive)' : 'var(--negative)' },
  ];

  document.getElementById('hbt-metrics-grid').innerHTML =
    metrics.map(m => `
      <div class="hbt-metric">
        <div class="hbt-metric-lbl">${m.l}</div>
        <div class="hbt-metric-val" style="${m.c ? 'color:' + m.c : ''}">${m.v}</div>
      </div>`).join('');

  // ── Equity curve chart ─────────────────────────────────────────────────
  if (_hbtChart) { _hbtChart.destroy(); _hbtChart = null; }
  const ctx = document.getElementById('hbt-chart').getContext('2d');
  _hbtChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ec.dates || [],
      datasets: [
        {
          label:       'Bot Strategy',
          data:        ec.portfolio_value || ec.values || [],
          borderColor: '#0dcfcf',
          borderWidth: 2,
          pointRadius: 0,
          fill:        false,
          tension:     0.2,
        },
        {
          label:       'Buy & Hold',
          data:        ec.buy_hold_value || [],
          borderColor: '#475569',
          borderWidth: 1.5,
          borderDash:  [4, 4],
          pointRadius: 0,
          fill:        false,
          tension:     0.2,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 12 } } },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.dataset.label}: $${ctx.parsed.y.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2})}`,
          },
        },
      },
      scales: {
        x: { ticks: { color: '#475569', maxTicksLimit: 8 }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#94a3b8', callback: v => '$' + (v/1000).toFixed(0)+'k' }, grid: { color: 'rgba(255,255,255,0.06)' } },
      },
    },
  });

  // ── Drawdown chart ─────────────────────────────────────────────────────
  if (_hbtDdChart) { _hbtDdChart.destroy(); _hbtDdChart = null; }
  const ddCtx = document.getElementById('hbt-dd-chart').getContext('2d');
  _hbtDdChart = new Chart(ddCtx, {
    type: 'line',
    data: {
      labels: ec.dates || [],
      datasets: [{
        label:           'Drawdown',
        data:            ec.drawdown || [],
        borderColor:     'rgba(239,68,68,0.7)',
        backgroundColor: 'rgba(239,68,68,0.08)',
        borderWidth:     1.5,
        pointRadius:     0,
        fill:            true,
        tension:         0.2,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#475569', maxTicksLimit: 6 }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#94a3b8', callback: v => v.toFixed(1) + '%' }, grid: { color: 'rgba(255,255,255,0.06)' } },
      },
    },
  });

  // ── Signal stats ───────────────────────────────────────────────────────
  document.getElementById('hbt-signal-stats').innerHTML = `
    <div class="hbt-sig-stat"><span class="bt-metric-label">SEÑALES TOTALES</span><span class="bt-metric-val">${sig.total_signals||0}</span></div>
    <div class="hbt-sig-stat"><span class="bt-metric-label">OPERADAS (top ${((1-(sig.filter_percentile||0.75))*100).toFixed(0)}%)</span><span class="bt-metric-val">${sig.filtered_signals||0}</span></div>
    <div class="hbt-sig-stat"><span class="bt-metric-label">UMBRAL SEÑAL</span><span class="bt-metric-val">${(sig.threshold||0).toFixed(4)}</span></div>
    <div class="hbt-sig-stat"><span class="bt-metric-label">SEÑAL MÁXIMA</span><span class="bt-metric-val">${(sig.max_signal||0).toFixed(4)}</span></div>`;

  // ── Per-asset breakdown ─────────────────────────────────────────────────
  const perAsset     = data.per_asset || [];
  const perAssetWrap = document.getElementById('hbt-per-asset-wrap');
  const perAssetGrid = document.getElementById('hbt-per-asset-grid');
  if (perAsset.length > 0 && perAssetGrid) {
    perAssetWrap.style.display = 'block';
    perAssetGrid.innerHTML = perAsset.map(a => {
      const pos   = a.pnl >= 0;
      const color = pos ? 'var(--positive)' : 'var(--negative)';
      return `<div class="hbt-asset-card">
        <div class="hbt-asset-sym">${a.symbol}</div>
        <div class="hbt-asset-pnl" style="color:${color}">${pos?'+':'-'}$${fmtNum(Math.abs(a.pnl))}</div>
        <div class="hbt-asset-meta">${a.trades} trades · ${a.win_rate}% wins</div>
      </div>`;
    }).join('');
  }

  // ── Trades table ───────────────────────────────────────────────────────
  const trades = data.trades || [];
  const empty  = document.getElementById('hbt-trades-empty');
  const wrap   = document.getElementById('hbt-trades-wrap');
  const tbody  = document.getElementById('hbt-trades-tbody');

  if (trades.length === 0) {
    empty.style.display = 'block';
    wrap.style.display  = 'none';
  } else {
    empty.style.display = 'none';
    wrap.style.display  = 'block';
    tbody.innerHTML = trades.map(t => {
      const pnl     = t.pnl || 0;
      const color   = pnl >= 0 ? 'var(--positive)' : 'var(--negative)';
      const ar      = t.actual_return_pct;
      const arColor = ar >= 0 ? 'var(--positive)' : 'var(--negative)';
      return `<tr>
        <td><strong>${t.symbol||'—'}</strong></td>
        <td>${t.entry_date || '—'}</td>
        <td>${t.exit_date  || '—'}</td>
        <td><span class="action-badge ${(t.action||'').toLowerCase()}">${t.action||'—'}</span></td>
        <td>$${fmtNum(t.entry_price)}</td>
        <td>$${fmtNum(t.exit_price)}</td>
        <td>${t.hold_days||0}d</td>
        <td style="color:${arColor}">${ar != null ? (ar>=0?'+':'')+ar.toFixed(2)+'%' : '—'}</td>
        <td style="color:${color};font-weight:600">${pnl>=0?'+':'-'}$${fmtNum(Math.abs(pnl))}</td>
        <td><span class="reason-badge">${t.exit_reason||'—'}</span></td>
      </tr>`;
    }).join('');
  }
}

// ══════════════════════════════════════════════
//  SIGNAL SCANNER
// ══════════════════════════════════════════════

async function scanRun() {
  const btn = document.getElementById('scan-run-btn');
  btn.disabled = true;
  btn.textContent = 'Escaneando...';
  document.getElementById('scan-loading').style.display  = 'flex';
  document.getElementById('scan-results').style.display  = 'none';
  document.getElementById('scan-error').style.display    = 'none';

  try {
    const res  = await fetch(`${API}/api/paper/scan`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ asset_type: _scanType, extra_symbols: _scanExtraSyms }),
    });
    const data = await res.json();
    if (data.status !== 'success') throw new Error(data.error || 'Error en scanner');

    scanRenderResults(data);
    document.getElementById('scan-results').style.display = 'block';
  } catch (err) {
    document.getElementById('scan-error').style.display   = 'block';
    document.getElementById('scan-error').textContent     = err.message;
  } finally {
    document.getElementById('scan-loading').style.display = 'none';
    btn.disabled    = false;
    btn.textContent = 'Escanear Mercado';
  }
}

function scanRenderResults(data) {
  const ts = new Date(data.timestamp).toLocaleTimeString('es-ES');
  document.getElementById('scan-timestamp').textContent =
    `Escaneado el ${new Date(data.timestamp).toLocaleDateString('es-ES')} a las ${ts} · ${data.scanned} activos`;

  const signals = data.signals || [];
  const tbody   = document.getElementById('scan-tbody');

  tbody.innerHTML = signals.map(s => {
    if (s.action === 'ERROR') {
      return `<tr class="scan-row-error">
        <td><strong>${s.symbol}</strong></td>
        <td colspan="7" style="color:var(--text-muted);font-size:0.8rem">Error: ${s.error}</td>
      </tr>`;
    }
    const pr      = s.predicted_return_pct || 0;
    const prColor = pr >= 0 ? 'var(--positive)' : 'var(--negative)';
    const ssBar   = Math.min((s.signal_strength / 0.02) * 100, 100).toFixed(0);
    const canExec = s.action !== 'HOLD' && !s.already_open;
    const alreadyOpen = s.already_open
      ? '<span style="font-size:0.7rem;color:var(--warning)">Posición abierta</span>' : '';

    return `<tr class="scan-row ${s.action.toLowerCase()}">
      <td>
        <strong>${s.symbol}</strong>
        <span style="font-size:0.72rem;color:var(--text-muted);display:block">${s.asset_type}</span>
      </td>
      <td><span class="action-badge ${s.action.toLowerCase()}">${s.action}</span></td>
      <td style="color:${prColor};font-weight:600">${pr>=0?'+':''}${pr.toFixed(2)}%</td>
      <td>${(s.confidence*100).toFixed(0)}%</td>
      <td>
        <div class="scan-ss-bar-wrap">
          <div class="scan-ss-bar" style="width:${ssBar}%"></div>
          <span>${s.signal_strength.toFixed(4)}</span>
        </div>
      </td>
      <td>${s.current_price != null ? '$'+fmtNum(s.current_price) : '—'}</td>
      <td>${s.suggested_size_pct}%</td>
      <td>
        ${alreadyOpen}
        ${canExec ? `
          <button class="execute-btn" style="padding:0.3rem 0.6rem;font-size:0.75rem;width:auto"
            onclick="scanExecute('${s.symbol}','${s.asset_type}','${s.action}',${s.suggested_size_pct},${s.signal_strength},${pr})">
            ${s.action}
          </button>` : ''}
      </td>
    </tr>`;
  }).join('');
}

async function scanExecute(symbol, assetType, action, sizePct, signalStrength, predReturn) {
  if (!confirm(`¿Ejecutar ${action} en ${symbol} con ${sizePct}% del capital?`)) return;

  try {
    const res  = await fetch(`${API}/api/paper/execute`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        symbol, asset_type: assetType, action,
        size_pct: sizePct, signal_strength: signalStrength,
        predicted_return_pct: predReturn, reason: 'SCANNER',
      }),
    });
    const data = await res.json();
    if (data.status !== 'success') throw new Error(data.error);

    // Refresh portfolio overview
    botRenderOverview(data.portfolio);
    botRenderPositions(data.portfolio.positions_with_pnl || []);
    botRenderHistory(data.portfolio);

    const t = data.trade;
    botShowToast(`${t.action} ${t.symbol} ejecutado @ $${fmtNum(t.entry_price)}`, 'success');

    // Re-run scan to refresh table
    scanRun();
  } catch (err) {
    botShowToast('Error: ' + err.message, 'error');
  }
}
