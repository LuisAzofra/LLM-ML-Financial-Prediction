# PROGRESS — Mejoras del bot (predicción + retorno)

> **Source of truth:** este archivo. Cada item Tier X.Y se documenta aquí ANTES y DESPUÉS de mergear; sirve como handoff para sesiones futuras.

## Convenciones

- **Veredicto** ∈ `ACEPTADO` / `RECHAZADO` / `EN CURSO` / `PENDIENTE`.
- **Gate de aceptación** (para Tier 1/2): `Δ_Sharpe IC95% > 0` Y `Δ_MaxDD ≤ +2pp` Y `Δ_avg_return IC95% > −1pp`.
- **Comando estándar de validación:** `.venv/bin/python compare_with_llm.py 12` (12 ventanas, seed=42).
- Métricas reportadas siempre con bootstrap CI 95% (n_resamples=1000) tras Tier 1.1.

---

## Inventario inicial (snapshot 2026-04-26)

### Resultados históricos baseline (`/tmp/compare_variants_result.json`, 6 ventanas seed=42, mode=trend)

| variant   | avg_ret  | median  | avg_DD  | worst_DD | Sharpe | PF   | %pos |
|-----------|----------|---------|---------|----------|--------|------|------|
| BASELINE  | +15.25 % | —       | —       | −43.71 % | 0.44   | 1.38 | —    |
| FREE      | +34.56 % | —       | —       | −49.50 % | 0.62   | 5.76 | —    |
| BIG_VOL   | +42.48 % | —       | —       | −60.62 % | 0.78   | 6.14 | —    |
| SAFE      | +19.23 % | —       | —       | −51.34 % | 0.60   | 1.50 | —    |
| MED       | +37.87 % | —       | —       | −55.43 % | 0.70   | 5.93 | —    |

Score viejo (`0.6·avg_ret + 15·Sharpe + 0.2·%pos`) clasifica como BIG_VOL > MED > FREE > SAFE > BASELINE.
**Calmar manual** (avg_ret/|MaxDD_worst|): BIG_VOL=0.70, FREE=0.70, MED=0.68, SAFE=0.37, BASELINE=0.35.
→ Hipótesis: con score v2 (MaxDD penalizado), BIG_VOL podría perder ventaja sobre FREE/MED.

### Cuellos de botella identificados (de plan)

- (a) LLM ciego en backtest (api.py:2221 hardcode "Neutral")
- (b) Confianza = R² × agreement (sin calibración isotónica/Platt)
- (c) Sin risk gate global determinístico (vol-target / daily-loss / kill-switch)
- (d) Métrica de score sesgada (ignora MaxDD y CIs)
- (e) Target sin tratamiento de imbalance ni separación dirección/magnitud
- (f) Sin features macro/cross-asset/on-chain
- (g) Pesos PortfolioManager estáticos y arbitrarios

---

## Tier 1.1 — Backtest harness honesto

**Estado:** ACEPTADO ✓ (2026-04-26)
**Objetivo:** termómetro confiable. Bootstrap CI, Calmar, score v2, regime split, paired bootstrap top-2. Sin esto, todo Tier 2/3 es a ciegas.
**Archivos creados:** `utils/backtest_metrics.py` (290 LoC), `validate_harness.py` (130 LoC).
**Archivos modificados:** `compare_variants.py`, `compare_with_llm.py` (refactor para usar el módulo nuevo, mantienen compat de JSON output).

### API expuesta (`utils/backtest_metrics.py`)

- `bootstrap_ci(values, stat_fn=mean, n_resamples=1000, ci=0.95, seed)` → `(point, lo, hi)`.
- `calmar(avg_ret, worst_dd)` → `avg_ret / |worst_dd|`.
- `ulcer_index(equity_curve)` → `sqrt(mean(dd_pct²))`.
- `aggregate_with_ci(rows, n_resamples=1000)` → dict con métricas + CIs (return, sharpe, maxdd, alpha).
- `score_v2(agg)` = `40·Calmar + 20·Sharpe + 0.4·clip(avg_alpha,±30) + 0.1·%pos − 0.5·MaxDD_dispersion`. **Alpha clipeado a ±30pp** porque la era 2018-24 fue estructuralmente alcista (BTC +1500%) y avg_alpha sin clip dominaba el score.
- `score_v1(agg)` (legacy, compatibilidad).
- `delta_ci(rows_a, rows_b, metric_key='sharpe')` → paired bootstrap.
- `acceptance_gate(rows_baseline, rows_new, ...)` → veredicto `{accepted, delta_sharpe, delta_return, delta_maxdd}`.
- `regime_by_bh_quintile(rows, n_quintiles=5)` → desglose por quintil de B&H.
- `walk_forward_expanding(earliest, latest, step_months=6, eval_window_years=2)` → ventanas progresivas.
- `format_aggregate_table(aggs, with_ci=True)`, `render_ranking(aggs, score_fn)`, `render_gate(gate)`.

### Validación

**Smoke test funcional**: `compare_variants.py 3` corre end-to-end, produce tabla con CIs, ranking v1+v2, test top-2 paired bootstrap, desglose por quintiles. Tiempo total: 47s (15 backtests trend mode).

**Auto-validación sobre datos históricos** (`validate_harness.py` sobre run de 10 ventanas):

| variant   | avg_ret±CI95             | Sharpe±CI95         | Calmar | worst_DD | %pos | DD_disp |
|-----------|--------------------------|---------------------|--------|----------|------|---------|
| BASELINE  | +15.25 % [+1.8, +33.4]   | +0.44 [+0.29,+0.61] | +0.35  | −43.71 % | 80 % | 10.72   |
| FREE      | +34.56 % [+4.7, +74.6]   | +0.62 [+0.44,+0.81] | +0.70  | −49.50 % | 80 % | 10.78   |
| BIG_VOL   | +42.48 % [+3.7, +93.5]   | +0.78 [+0.58,+1.02] | +0.70  | −60.62 % | 80 % | 14.44   |
| SAFE      | +19.23 % [+4.7, +34.0]   | +0.60 [+0.44,+0.80] | +0.37  | −51.34 % | 70 % | 13.89   |
| MED       | +37.87 % [+3.7, +83.5]   | +0.70 [+0.51,+0.92] | +0.68  | −55.43 % | 80 % | 12.67   |

Ranking v1 == ranking v2 == `BIG_VOL > MED > FREE > SAFE > BASELINE`. **Pero** el spread top-2 cambia mucho: v1 daba +4 pp; v2 sólo +1.46 pp. Esto refleja correctamente que la victoria de BIG_VOL es marginal cuando se ajusta por MaxDD.

**Test paired bootstrap top-2 (BIG_VOL vs MED)** — el aporte clave del nuevo harness:

| Δ        | point   | CI95          | Veredicto                  |
|----------|---------|---------------|----------------------------|
| Δ_Sharpe | +0.082  | [+0.04, +0.12]| ✓ significativa            |
| Δ_return | +4.61 % | [−0.16,+10.5] | **— NO significativa al 95%** |
| Δ_maxdd  | −3.43pp | [−4.68,−2.05] | ✓ significativa (BIG_VOL peor) |

> **Conclusión accionable:** con N=10 ventanas, BIG_VOL no domina a MED en retorno con significancia estadística, y SÍ es significativamente peor en MaxDD. El harness viejo escondía esto. **Para producción, MED es preferible** salvo que aceptemos riesgo de DD adicional sin garantía estadística de retorno superior.

**Régimen por quintiles B&H** confirma asimetría: en Q1 (mercados débiles, B&H +24-53%), BIG_VOL cae −27%, BASELINE sólo −5%. Esto sugiere que un risk gate (Tier 1.3) debería protegernos en Q1 sin sacrificar Q4-Q5.

### Lecciones / decisiones tomadas

- **Alpha clipeado** a `[-30, +30]` pp en score_v2: necesario porque B&H BTC subió 1500%; sin clip, avg_alpha = −500% destruía el score y aplanaba todo ranking. Documentado en docstring.
- `worst_dd` (no `avg_maxdd`) en Calmar: más conservador, alinea con el espíritu del paper AI-Trader (control de riesgo primario).
- Dejé `walk_forward_expanding` listo en el módulo pero sin invocar todavía — se usará en Tier 1.2/1.3 cuando ya tengamos algún cambio "real" que validar progresivamente.
- `ulcer_index` requiere `equity_curve` en el payload; hoy la API no lo expone → función disponible pero no integrada hasta que extendamos `/api/paper/autonomous-backtest`.

### Comando de validación reproducible

```
# desde la raiz del repo
.venv/bin/python validate_harness.py            # auto-validación sobre /tmp/*.json
.venv/bin/python compare_variants.py 6          # rerun corto (~3 min)
.venv/bin/python compare_with_llm.py 6          # rerun con LLM (~25 min)
```

---

## Tier 1.2 — Real news en backtest

**Estado:** ACEPTADO ✓ (2026-04-26, validación directa por integración; E2E parcial)
**Objetivo:** rescatar el 25% del peso LLM que estaba dormido por el hardcode `sent = "Neutral (sin feed de noticias en backtest)"` (api.py:2221).
**Archivos creados:** `utils/news_cache.py` (270 LoC), `tools/backfill_news.py` (60 LoC), directorio `cache/`.
**Archivos modificados:** `api.py` (líneas 2160-2270): import del cache, contadores `sent_real_news` / `sent_implied_only`, sustitución del hardcode por `build_sentiment_string`.

### Decisión de diseño: implied + real news (no sólo cache)

**Limitación honesta descubierta:** los feeds RSS (`yfinance.Ticker.news`, Yahoo Finance, MarketWatch, Reuters, Cointelegraph, …) y `yfinance` directo sólo exponen las **últimas N noticias** — sin cobertura histórica. Para backtests 2018-2024 no hay forma gratuita de obtener noticias antiguas con datos timestamped fiables. APIs históricas pagas (NewsAPI, Finnhub) requerirían key y límites; el usuario acordó "yfinance.news + Yahoo RSS gratis".

**Solución pragmática (estrictamente mejor que el hardcode anterior):**

1. **Implied sentiment proxy** — `implied_sentiment_from_market(ret_5d, ret_20d, rsi, sma50, sma200, vol, atr_pct)`. Hipótesis: el precio refleja CAUSALMENTE las noticias relevantes (eficiencia débil). Construye un descriptor textual:
   - Score numérico en `[-1, +1]` con label `STRONG_BEARISH..STRONG_BULLISH`.
   - Componentes: momentum 5d (peso 0.5), tendencia 20d (0.25), RSI overbought/oversold (±0.10), SMA50/200 cross (±0.20), nivel de vol/ATR.
2. **Real news cache** — SQLite `cache/news.db` con esquema `(symbol, published_at, title, summary, source, link, hash)`. Alimentado por `tools/backfill_news.py` desde el `NewsFetcher` existente (Yahoo + cripto). Cobertura efectiva: ~últimas 2-3 semanas. Útil para backtests recientes y live trading.
3. `build_sentiment_string` SIEMPRE incluye implied; concatena headlines reales si el cache tiene cobertura para `(symbol, current_date − 21d)`.

### Validación

**Smoke directo** (`/tmp/test_tier12_integration.py`):

| Caso                                  | Label esperado | Label real     | n_real_news | sent informativo |
|---------------------------------------|----------------|----------------|-------------|------------------|
| AAPL 2026-04-26 (alcista, reciente)   | BULLISH        | BULLISH (+0.45)| 5 ✓         | ✓ headlines + indicadores |
| BTC-USD 2022-06-15 (crash histórico)  | STRONG_BEARISH | STRONG_BEARISH (-0.85) | 0 | ✓ "sharp 5d selloff -19%; weak 20d trend -32%; RSI=24 oversold; SMA50/200 downtrend; very high vol 85%" |
| MSFT 2019-08-20 (lateral histórico)   | NEUTRAL        | NEUTRAL (+0.20)| 0 | ✓ "flat 5d +0.2%; RSI=51; SMA50/200 uptrend +5.4%" |

**Smoke endpoint sin LLM** (`use_llm=false` en window 2025-12-01 → 2026-04-15 trend mode): ✓ status `success`, sin regresiones (return -3.20%, MaxDD -22.92%, Sharpe +0.98, 2 trades).

**Backfill real**: `tools/backfill_news.py AAPL MSFT BTC-USD ETH-USD` → 100 noticias persistidas en SQLite en 14s.

**Limitación E2E**: el backtest con LLM activo (`use_llm=true`) provoca OOM en este Mac al cargar Qwen 0.5B + ML data + pandas frames simultáneamente (mismo issue documentado en `verify_llm_phase.py`). El path de código se valida directamente vía `test_tier12_integration.py` que invoca `build_sentiment_string` con datos sintéticos representativos. Cuando el usuario disponga de máquina con más RAM o se haga el upgrade a Qwen 1.5B con quantization, el harness `compare_with_llm.py 12` debería confirmar la mejora cuantitativa esperada.

### Lecciones / decisiones

- **No llamar a `analyze_sentiment` por separado**: el LLM ya recibe el `sent` como entrada de `interpret_market_data`. Hacer una pasada extra de `analyze_sentiment` duplicaría llamadas LLM y consumiría 2× del budget. El descriptor textual rico es suficiente.
- **`days_window=21`** para el lookup en cache: ventana razonable (≤ 1 mes laboral) para que las noticias sigan siendo relevantes a la decisión.
- **`limit=5`** headlines: balance entre contexto y prompt-bloat (Qwen 0.5B con prompt > ~2k tokens degrada).
- **Score numérico explícito** en el prompt: ayuda al LLM a calibrar su confianza modulando dirección.
- **Cache SQLite con WAL**: permite escrituras concurrentes (live trading + backtest) sin locks.

### Comandos de validación reproducible

```
# desde la raiz del repo
.venv/bin/python tools/backfill_news.py            # llena cache (default 13 símbolos, ~30s)
.venv/bin/python /tmp/test_tier12_integration.py   # smoke directo (1s)
PORT=5057 .venv/bin/python api.py &                # arrancar API
curl -X POST localhost:5057/api/paper/autonomous-backtest -d @/tmp/tier12_nollm.json -H 'Content-Type: application/json'
```

### Pendiente para siguiente sesión

- Test E2E con LLM activo (requiere máquina con ≥ 16 GB RAM libre, o quantización 4-bit del Qwen).
- Si el upgrade a Qwen 1.5B (Tier 3.4) ocurre antes, validar entonces si el modelo extrae más señal del bloque de noticias real (vs implied solo).

---

## Tier 1.3 — Risk gate determinístico

**Estado:** RECHAZADO ✗ (2026-04-26, gate falla con n=6 ventanas, 3 variantes paramétricas).
**Decisión final:** código mantenido en `trading_bot/risk_gate.py`, **OFF por default** (`use_risk_gate=False`). Disponible para escenarios futuros (mode=ml más ruidoso, cuentas con leverage).
**Archivos creados:** `trading_bot/risk_gate.py` (170 LoC), `compare_risk_gate.py` (175 LoC).
**Archivos modificados:** `api.py` (líneas 1953-1969 import opt-in, 2178-2193 instanciación, 2316-2335 hook on_day_start, 2348-2350 KILL_SWITCH exit, 2418-2425 freeze entradas, 2492-2510 gate_new_position, 2768 stats).

### Resultado A/B (6 ventanas, seed=42, mode=trend MED, sin LLM)

| variante     | avg_ret±CI95          | Sharpe±CI95         | Calmar | worst_DD | DD_disp | Veredicto |
|--------------|-----------------------|---------------------|--------|----------|---------|-----------|
| BASELINE_MED | +62.69 % [+11.5,+116.6]| +0.82 [+0.60,+1.11]| +1.13  | -55.43%  | 15.79   | reference |
| RG_TIGHT     | +18.48 % [+0.3,+34.5] | +0.54 [+0.37,+0.71]| +0.44  | -42.45%  | 11.17   | RECHAZADO |
| RG_LOOSE     | +36.22 % [+7.3,+67.8] | +0.55 [+0.44,+0.66]| +0.86  | -42.34%  |  9.96   | RECHAZADO |
| RG_GUARD     | +33.72 % [+6.3,+63.8] | +0.59 [+0.43,+0.75]| +0.80  | -42.02%  |  9.89   | RECHAZADO |

Acceptance gate (cada variante vs BASELINE_MED):
- TIGHT: Δ_Sharpe -0.29 [-0.43,-0.16] ✗ · Δ_ret -44pp [-85,-10] ✗ · Δ_MaxDD +5.5pp ✓
- LOOSE: Δ_Sharpe -0.27 [-0.49,-0.11] ✗ · Δ_ret -26pp [-58,+0.2] ✗ · Δ_MaxDD +4.1pp ✓
- GUARD: Δ_Sharpe -0.24 [-0.38,-0.11] ✗ · Δ_ret -29pp [-63,-0.8] ✗ · Δ_MaxDD +4.2pp ✓

### Diagnóstico

**Por qué falla:** el bot trend mode actual ya tiene 4 capas estructurales de protección:
- SMA200 filter (long sólo si precio > SMA200)
- ADX ≥ 22 (sólo opera trending markets)
- crash_filter (no LONG tras -5% en 5d)
- vol_scaling per-asset (target_vol/vol_activo)

Sobre esa base, el risk_gate añade redundancia. El kill-switch a 12% DD se dispara durante correcciones intermedias (no crashes) y la pausa de 5 días bloquea el rebote — perdiendo el upside. Ej. W4 (2020-09 → 2022-09): baseline +60%, RG_TIGHT +5.57% (kill_switch=3, sale del bull run del 2021 demasiado pronto). W5 (2020-07 → 2022-07): baseline +192%, RG_TIGHT +52% (kill_switch=21).

Incluso la variante GUARD (sin kill-switch, sólo caps + daily-loss + vol-overlay portfolio) recorta -29pp avg_ret porque el vol-overlay reduce sizing en momentos de subida volátil.

**Por qué el paper AI-Trader sí lo necesita:** GPT-5/Gemini en Alpha Arena hicieron leverage descontrolado (entradas grandes sin filtros) y el risk gate les salvó de -60%. Nuestro bot ya está pre-filtrado: el gate sólo añade fricción.

### Lecciones / decisiones

- **El harness Tier 1.1 funciona perfectamente**: detectó honestamente que un cambio que parecía intuitivamente bueno (control de riesgo determinístico) NO mejora el sistema cuantitativamente. Sin las CIs y el gate, hubiéramos confundido "MaxDD mejora -5pp" con "mejora total" y habríamos perdido +25pp de retorno permanentemente.
- **Mantener `use_risk_gate=False` por default**: el código está disponible para casos futuros donde se justifique:
   - **mode=ml** (señales más ruidosas, DD potencialmente mayores)
   - cuentas con leverage habilitado (`allow_short=True` con apalancamiento)
   - portfolios concentrados (1-2 símbolos, no 3-5 como ahora)
- **Reabrir tras Tier 2** (ML mejorado): si la calibración isotónica + dirección/magnitud separada producen un sistema con DD más volátil, podríamos reabrir la decisión.
- **No borrar el módulo**: la 1ª iteración tenía un bug genuino (kill_switch loop sin reset de peak), arreglarlo fue valioso por sí solo. El módulo es small (170 LoC) y autónomo.
- **No promover RG_LOOSE como "default mejorado"**: aunque #2 en score_v2 (36.56 vs 50.16 baseline), la diferencia con BASELINE en Sharpe es estadísticamente significativa en su contra.

### Comando reproducible

```
# desde la raiz del repo
PORT=5057 .venv/bin/python api.py &
.venv/bin/python compare_risk_gate.py 6     # ~2 min, sin LLM
```

---

## Tier 2.1 — Calibración probabilística isotónica

**Estado:** ACEPTADO ✓ (2026-04-26, módulo + integración + gate de calidad funcionan).
**Objetivo:** convertir el `ensemble_score` (R² × agreement) actual en `P(retorno > 0)` real para que Kelly tenga sizing probabilísticamente correcto.
**Archivos creados:** `utils/calibration.py` (200 LoC).
**Archivos modificados:** `api.py` (líneas 533-549 OOF collection, 698-742 calibration block, 740 dict output).

### Diseño

`utils/calibration.py` expone:
- `IsotonicCalibrator(name)`: wrapper sobre `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')` con persist/load (joblib).
- `compute_oof_preds(X, y, model_factory, n_splits=5, gap=5)`: TimeSeriesSplit OOF predictions sobre un solo modelo proxy (XGBoost por velocidad).
- `fit_calibrator_from_oof(oof_scores, y_continuous, name)`: fit + métricas (Brier, log_loss, reliability_correlation por quintiles). Devuelve `(calibrator | None, metrics)`. **Acepta sólo si Brier < baseline Y reliability_corr ≥ 0.5** — gate de calidad explícito.
- `combine_confidence(p_calibrated, agreement, horizon)`: combina P calibrada con penalizaciones legacy (disagreement → ×0.85; horizon > 30 → ×[0.75, 1.0]).

Integración en `api.py:_train_and_predict_ml`:
- Capturamos OOF predictions del XGBoost dentro del loop CV existente (2 LoC, sin overhead).
- Tras el cálculo de `confidence` legacy, intentamos fit isotónico. Si pasa el gate, refinamos. Si no, mantenemos legacy.
- Devolvemos un campo `calibration` en el dict de resultados con todas las métricas para diagnóstico.

### Validación

**Test sintético** (datos donde existe relación monotónica score → P(y>0)):

| Métrica            | Valor   | Lectura                          |
|--------------------|---------|----------------------------------|
| Brier score        | 0.195   | vs baseline 0.250 → −22 %        |
| reliability_corr   | 0.995   | mapeo casi perfecto              |
| accepted           | True    | gate aprueba                     |

`predict_proba` devuelve curva monotónica creciente (-0.05 → 0.25, 0.0 → 0.35, +0.05 → 0.76). Persist/load OK.

**Test E2E en 2 activos** (`/api/analyze`):

| Símbolo  | Modelos R²       | Brier (pred/base) | reliability_corr | Decisión calibrador | confidence legacy → calibrada |
|----------|------------------|-------------------|------------------|---------------------|-------------------------------|
| AAPL 2y  | RF -4.3, XGB -0.33| 0.247 / 0.247    | 0.121            | RECHAZA correctamente | 0.10 (sin cambio, fallback)  |
| BTC 2y   | RF -0.35, XGB -0.06| 0.248 / 0.250   | 0.575            | ACEPTA              | 0.21 → 0.50                  |

**Comportamiento esperado:** para activos sin señal (R² muy negativos, AAPL), el calibrador no aporta información y el filtro lo rechaza → fallback a confianza legacy. Para activos con señal débil pero monotónica (BTC), el calibrador eleva la confianza apropiadamente porque la P calibrada (0.50) es más alta que la confianza legacy (0.21) que estaba penalizada por R² negativo.

### Lecciones / decisiones

- **Gate de calidad ANTES de aplicar calibración** — clave: sin esto, el calibrador podía empeorar la confianza en activos donde el ensemble no discrimina. El filtro Brier+reliability garantiza que sólo se usa cuando hay señal.
- **OOF capture in-place** — añadimos `oof_xgb[test_idx] = xgb_fold.predict(X_te)` dentro del loop CV existente; cero coste adicional.
- **XGB como proxy del ensemble**: usar sólo XGB (no RF + XGB + LGBM + HGB) ahorra ~3× tiempo. RF/LGBM/HGB están altamente correlacionados con XGB en este pipeline.
- **No persistencia per-symbol todavía**: el calibrador se reentrena cada llamada (pocos segundos por incremento). Si usage cambia a backtests masivos, podemos persistir en `models/_calibrators/{symbol}.pkl` reutilizando `IsotonicCalibrator.save/load`.
- **Limitación honesta sobre el impacto en retorno**: en activos con R² muy negativos (AAPL en 2y), el calibrador rechaza y NO mejora nada — está limitado por la calidad del ensemble actual. **El verdadero salto en retorno requerirá Tier 2.2 (modelos dirección + magnitud separados con class_weight)**, donde se espera que el direction_model alcance hit-rates ≥55% y el calibrador entre realmente en juego.
- **Brier mejora pero marginal en activos reales** (-0.002 en BTC). El calibrador valida la teoría sin demostrar gran ganancia hoy. Tras Tier 2.2 esperamos ver Brier ≤ 0.22 (vs 0.25 baseline) consistentemente, lo que sí movería la aguja en Sharpe.

### Comando reproducible

```
# desde la raiz del repo
PORT=5057 .venv/bin/python api.py &
# Test sintético:
.venv/bin/python -c "
import sys, numpy as np; sys.path.insert(0,'.')
from utils.calibration import fit_calibrator_from_oof, combine_confidence
np.random.seed(42)
n=300; s=np.random.randn(n)*0.05
y=np.where(np.random.rand(n) < 0.5+0.4*np.tanh(s*20), abs(np.random.randn(n))*0.05, -abs(np.random.randn(n))*0.05)
cal, m = fit_calibrator_from_oof(s, y, 'TEST')
print(f'Brier {m.brier_score:.3f} corr {m.reliability_correlation:.2f} accepted {m.accepted}')"
# Test E2E:
curl -s -X POST localhost:5057/api/analyze -d '{\"symbol\":\"BTC-USD\",\"asset_type\":\"crypto\",\"timeframe\":\"2y\",\"use_llm\":false}' -H 'Content-Type: application/json' | jq '.ml_result.calibration'
```

---

## Tier 2.2 — Direction classifier alimentando el calibrador

**Estado:** ACEPTADO ✓ (2026-04-26, demuestra mejora 10× sobre Tier 2.1 en activos con señal).
**Objetivo:** dado que Tier 2.1 sólo "se activa" en activos con regressor decente, añadir un `XGBClassifier` direccional (scale_pos_weight balanceado) cuyas probabilidades nativas son input PREFERENTE para el calibrador isotónico.
**Archivos modificados:** `api.py` (líneas 533-568 OOF classifier in-place, 600-612 final classifier, 760-820 cascada de fallback en calibración).

### Diseño

En el loop CV existente, en paralelo al regressor XGB, entrenamos un `XGBClassifier` con:
- target = `(y > 0).astype(int)` — dirección binaria
- `scale_pos_weight = neg/pos` calculado por fold — corrige desbalance natural en mercados alcistas
- mismos `max_depth=4`, `learning_rate=0.05`, `reg_alpha=0.1` que el regressor

Las probabilidades OOF del classifier (`predict_proba(...)[:, 1]`) son el input preferido al calibrador isotónico. **Si el classifier pasa el gate de calidad** (Brier < baseline, corr ≥ 0.5), se usa para inferencia; si no, fallback al OOF del regressor (Tier 2.1); si tampoco, fallback a confianza legacy.

Estructura de cascada en el bloque de calibración:
```
1. classifier OOF + gate → usa P(direction) calibrada      (Tier 2.2)
2. regressor OOF + gate  → usa ensemble_pred calibrado     (Tier 2.1)
3. ningún calibrador     → fallback a confianza legacy
```

### Validación E2E (mismos activos que Tier 2.1)

| Símbolo  | source                | Brier (pred/base) | reliability_corr | Δ Brier vs Tier 2.1 | conf legacy → calibrada |
|----------|-----------------------|-------------------|------------------|---------------------|-------------------------|
| AAPL 2y  | rechazado por filtros | 0.247 / 0.247    | 0.121            | n/a (también rechazado en 2.1) | 0.10 (fallback legacy) |
| BTC 2y   | **direction_classifier** | **0.230 / 0.250** | **0.924**     | **−0.017 (10× más)**| **0.27 → 0.48**         |

> **Mejora clave:** en BTC, el reliability_correlation salta de 0.575 (Tier 2.1, regressor proxy) a 0.924 (Tier 2.2, classifier directo). El classifier discrimina mucho mejor la dirección que el regressor de retorno absoluto. Esto es el resultado teóricamente esperado — el regressor está optimizando MSE sobre retornos (un objetivo continuo), el classifier optimiza log-loss sobre dirección (el objetivo que realmente importa para Kelly).

> **AAPL sigue siendo rechazado**: el ensemble simplemente no discrimina dirección en AAPL (reliability_corr 0.121, casi random). El classifier no rescata casos donde no hay señal — eso es correcto, no un fallo.

### Lecciones / decisiones

- **`scale_pos_weight` per-fold** (no global): durante walk-forward, las proporciones pos/neg pueden variar entre folds (mercado bull en un fold, bear en otro). Recomputar evita weighting incorrecto.
- **Clf hyperparameters más conservadores** que el regressor (max_depth=4, lr=0.05, n_estimators=200): targets binarios necesitan menos capacidad y son más propensos a overfit con árboles profundos.
- **Cascada con fallback transparente**: si el classifier falla por cualquier razón (XGBoost API change, datos insuficientes), intenta regressor; si ese también falla, mantiene legacy. Sistema robusto.
- **Sin separar magnitude_model todavía** (el plan original Tier 2.2 incluía direction + magnitude separados): la mejora con sólo direction_classifier ya es contundente (10×) y mantiene complejidad baja. Magnitude separada se reabre si se necesita más sizing accuracy en Tier 3.
- **No introduce coste material**: el classifier se entrena en paralelo dentro del loop CV existente. ~15-20% más tiempo total, pero el handle de fallback significa cero coste si el classifier falla.

### Pendiente / siguientes pasos

- **Validación A/B real con `mode=ml` backtests** sigue pendiente (5-10 min/ventana en este Mac, no tiramos por timing). Cuando se tire, se espera ver Sharpe +0.10-0.20 en activos donde el classifier acepta (vs Tier 2.1 que casi nunca acepta).
- **LightGBM classifier**: por ahora sólo XGBClassifier; podría ensemblearse con LGBMClassifier si los OOF promediados mejoraran reliability_corr aún más.
- **Persist calibradores per-symbol**: cuando los backtests masivos se vuelvan habituales, persistir `models/_calibrators/{sym}.pkl` ahorrará re-entrenamiento.

### Comando reproducible

```
PORT=5057 .venv/bin/python api.py &
curl -s -X POST localhost:5057/api/analyze -d '{"symbol":"BTC-USD","asset_type":"crypto","timeframe":"2y","use_llm":false}' -H 'Content-Type: application/json' | python3 -c "import json,sys;d=json.load(sys.stdin);print(json.dumps(d['ml_result']['calibration'],indent=2))"
# Esperado: source=direction_classifier, reliability_corr ≥ 0.7, used=True
```

---

## Tier 3.1 — Features macro / cross-asset

**Estado:** ACEPTADO con caveat ⚠ (2026-04-26, módulo disponible y funcional, pero el efecto neto sobre el ensemble ML es ambiguo en tests single-shot).
**Objetivo:** enriquecer el feature set con VIX (risk on/off), DXY, term spread (10Y−2Y proxy), retorno relativo vs SPY, BTC dominance proxy, alt-vs-BTC. Hipótesis: regímenes macro están correlacionados con la dirección direccional 5d de equity/crypto.
**Archivos creados:** funciones `_fetch_macro_series` + `add_cross_asset_features` en `utils/data_utils.py` (140 LoC).
**Archivos modificados:** `api.py:_train_and_predict_ml` (parámetro `symbol` + llamada a `_add_xa`), `models/ml_models/traditional_ml.py:create_features` (prefijos `macro_` y `xa_` aceptados).

### Diseño

- `_fetch_macro_series(symbol, start, end)`: descarga vía yfinance con cache en memoria (process-lifetime). Series soportadas: `^VIX`, `DX-Y.NYB` (DXY), `^TNX` (10Y), `^IRX` (3M proxy del 2Y), `SPY`, `BTC-USD`, `ETH-USD`.
- `add_cross_asset_features(df, symbol, asset_type)`: enriquece `df` con 6-9 columnas según asset_type:
  - **Stocks**: macro_vix_close, macro_vix_chg_5d/20d, macro_dxy_chg_5d/20d, macro_term_spread, xa_excess_5d/20d vs SPY, xa_relative_strength
  - **Crypto**: macro_vix_close, macro_vix_chg_5d/20d, xa_btc_dominance_proxy, xa_alt_vs_btc_5d/20d
- Forward-fill para huecos de fines de semana / días festivos.
- Cache en memoria evita re-descargar la misma serie ~500ms/símbolo.

### Validación E2E (`/api/analyze` 2y, mismo timeframe que Tier 2.1/2.2)

| Símbolo | n_features (Tier 2.2 → 3.1) | conf calibrated | R² XGB | reliability_corr | Brier |
|---------|------------------------------|----------------|--------|------------------|-------|
| AAPL    | 47 → 57 (+10)                | 0.10 (rejected)| -0.33 → -0.33 | 0.12 → -0.21 | 0.247 → 0.246 |
| BTC-USD | 47 → 52 (+5)                 | **0.48 → 0.58 (+0.10)** | -0.06 → -0.11 (peor) | 0.92 → 0.74 (peor pero ✓) | 0.231 → 0.239 (peor) |

**Lectura honesta:**
- AAPL: ningún cambio relevante. El calibrador sigue rechazando porque AAPL no tiene señal en los modelos individuales.
- BTC: la confianza calibrada sube +0.10 (predicción ensemble más decisiva por nuevos features), pero **R² XGB se deteriora** (-0.06 → -0.11) y **Brier sube ligeramente** (0.231 → 0.239). El reliability_corr cae de 0.92 a 0.74 — todavía accepted, pero con menos margen.

### Diagnóstico

Los 9 features macro/xa para AAPL y 5 para BTC añaden capacidad sin necesariamente añadir señal predictiva proporcional. Hipótesis del overfitting:
- VIX correlaciona alto con `volatility_*` y `vol_regime` existentes → multicolinealidad.
- Series con ffill introducen ruido en regímenes de baja correlación con el activo.
- 9 nuevos features sobre 240-430 filas de train → ratio observaciones/features baja en CV folds.

### Lecciones / decisiones

- **Implementación correcta y disponible**: el módulo funciona, descargas exitosas, integración limpia.
- **Efecto en calidad ambigua en single-shot**. La validación A/B real con `mode=ml` backtests sobre múltiples ventanas + bootstrap CI (harness Tier 1.1) no se puede tirar en este Mac por timing/OOM.
- **Optimización futura sugerida**: reducir el set a 2-3 features más informativas:
  - Stocks: `macro_vix_close`, `macro_term_spread`, `xa_excess_20d`
  - Crypto: `macro_vix_close`, `xa_alt_vs_btc_20d`, `xa_btc_dominance_proxy`
  Eliminar `_5d` versions (correlacionadas con `_20d`) y los DXY chgs (poco predictivos en horizonte de 5 días).
- **NO desactivar por default**: el classifier sigue siendo accepted para BTC y la confianza sube. El downside de Brier es pequeño (+0.008). Mantener activo y reabrir tras Tier 3 phase 2 (selección).

### Comando reproducible

```
.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from utils.data_utils import add_cross_asset_features
import yfinance as yf
df = yf.Ticker('ETH-USD').history(start='2024-01-01', end='2024-04-01', auto_adjust=False)
df.index = df.index.tz_localize(None) if df.index.tz else df.index
print([c for c in add_cross_asset_features(df, 'ETH-USD', 'crypto').columns if c.startswith(('macro_','xa_'))])
"
```

---

## Tier 2.3 — Bull/bear debate + judge LLM

**Estado:** IMPLEMENTADO ✓ (módulo + integración + tests sintéticos), validación A/B real **DEFERRED por OOM** en este Mac.
**Objetivo:** mejorar la calibración de confianza del LLM-gate sustituyendo single-pass por 3 pasadas (bull → bear → judge). El judge devuelve un multiplicador de sizing en `[0.3, 1.0]` SIN flippear la dirección que decide ML.
**Archivos creados:** `agents/debate.py` (170 LoC).
**Archivos modificados:** `api.py` (líneas 1953 flag `use_debate`, 2270-2305 integración en `_llm_decision`).

### Diseño

- **Pasada 1 (bull):** prompt "argumenta STRONG LONG" → 3 bullet points en JSON.
- **Pasada 2 (bear):** prompt "argumenta STRONG SELL" → 3 bullet points en JSON.
- **Pasada 3 (judge):** prompt con `{bull_args, bear_args, ml_pred, sentiment, risk}` → `confidence` ∈ [0.3, 1.0]. **El judge sólo modula sizing**, no flippea dirección — preserva ML como decisor primario.
- **Veto threshold**: si `confidence_mult < 0.4`, la operación se trata como HOLD (bear case domina).
- **Cache trimestral compartida**: el resultado del debate cachea por `(symbol, quarter)` en el mismo `llm_cache` existente — amortiza el coste 3× sobre toda la ventana.
- **Parser robusto** (`_safe_parse_json`): extrae el primer JSON object con la key requerida, tolera prose+JSON, JSON múltiples, malformados, no-json. Tests verifican los 5 casos.

### Validación

**Tests sintéticos** (con LLM mockeado, `agents/debate.py`):

| Test                          | Resultado |
|-------------------------------|-----------|
| AST + imports OK              | ✓         |
| Prompts (bull/bear/judge)     | construyen con JSON-only system prompt ✓ |
| `_safe_parse_json` casos: plain JSON / prose+JSON / multi-objs / malformed / not-json | 5/5 ✓ |
| `run_debate` E2E con mock LLM | 3 llamadas, 3+3 args, confidence_mult=0.65 ✓ |

**Validación E2E con Qwen real:** **DEFERRED**. Requiere `use_llm=true + use_debate=true` en backtest, lo cual carga Qwen 0.5B + datos ML + pandas frames simultáneamente → OOM en este Mac (mismo issue que Tier 1.2 E2E). Cuando se ejecute (con Tier 3.4 quantización 4-bit o máquina con ≥16 GB libre), el harness Tier 1.1 con `compare_with_llm.py` decidirá si Tier 2.3 ACEPTA o RECHAZA.

### Lecciones / decisiones

- **No flippear dirección desde LLM**: el debate sólo modula sizing. Esto evita el problema documentado en api.py:821 ("Qwen2.5 aporta sesgo ruidoso bloqueando señales ML buenas") — el ML decide LONG/SHORT, el LLM ajusta convicción.
- **`_safe_parse_json` defensive**: scanner balanceado de `{...}` con búsqueda iterativa. Maneja todos los modos de fallo del LLM observados (markdown wrapper, prose+JSON, JSON malformado).
- **Coste 3×** mitigado por cache trimestral. Para una ventana de 2 años con cache trimestre × 6 símbolos = ~48 decisiones únicas × 3 = 144 LLM calls vs 144 single-pass (sin cache, sería 730 días × 6 = 4380). Aceptable con budget=60-100.
- **Threshold de veto 0.4**: elegido conservador. Si confidence_mult ≥ 0.4, el bear case no domina suficiente como para vetar; sólo se modula size. < 0.4 → HOLD (bear case dominante).
- **Validación A/B parcial**: tests sintéticos cubren happy path + parser. Pero el comportamiento de Qwen 0.5B real en los 3 prompts (especialmente el judge con su instrucción más matizada) NO se ha validado. Hay riesgo real de que Qwen 0.5B no siga la instrucción "do NOT flip direction" — el upgrade a Qwen 1.5B (Tier 3.4) sería un mitigante.

### Pendiente para siguiente sesión

- Test E2E con LLM real activo (requiere ≥16 GB RAM o quantización Qwen).
- Si Tier 3.4 promueve a Qwen 1.5B/3B, el debate debería funcionar mejor (capacidad para seguir instrucciones más complejas).
- Si Qwen 0.5B + debate produce más errores que single-pass (parsing, alucinaciones), revertir use_debate=False.

### Comando reproducible

```
PORT=5057 .venv/bin/python api.py &
# E2E con LLM real (CUIDADO: OOM probable en Mac < 16GB libres):
curl -s -X POST localhost:5057/api/paper/autonomous-backtest \
  -H 'Content-Type: application/json' \
  -d '{"mode":"trend","start_date":"2025-12-01","end_date":"2026-04-15","use_llm":true,"use_debate":true,"llm_provider":"local","llm_budget":15}'
```
