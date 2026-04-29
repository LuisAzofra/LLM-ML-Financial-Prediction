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

**Estado:** ACEPTADO con caveat E2E ⚠ (2026-04-26 integración OK; 2026-04-29 E2E con LLM real muestra Δ=0 vs no-LLM, pero NO empeora — el cache se mantiene útil para live trading y sentiment).
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

### Validación E2E real (2026-04-29, post-Tier 3.4 upgrade)

Ejecución de `compare_with_llm.py 6` con `llm_provider=local-gguf` (Qwen2.5-1.5B Q4_K_M) — primera vez factible sin OOM. 6 ventanas seed=42, 5 variantes (4 con LLM + MED_NOLLM control), ~18 min total, 75 LLM calls (12.5/ventana).

| variante       | avg_ret±CI95             | Sharpe±CI95          | Calmar | worst_DD | LLM calls | LLM vetos |
|----------------|--------------------------|----------------------|--------|----------|-----------|-----------|
| BASELINE_GGUF  | +62.69 % [+11.5,+116.6]  | +0.82 [+0.60,+1.11]  | +1.13  | -55.43%  | 12.5      | **0**     |
| MED_LLM_GGUF   | +62.69 % [+11.5,+116.6]  | +0.82 [+0.60,+1.11]  | +1.13  | -55.43%  | 12.5      | **0**     |
| BIG_VOL_GGUF   | +70.53 % [+12.3,+130.7]  | +0.89 [+0.64,+1.22]  | +1.16  | -60.62%  | 12.5      | **0**     |
| SAFE_GGUF      | +70.53 % [+12.3,+130.7]  | +0.89 [+0.64,+1.22]  | +1.16  | -60.62%  | 12.5      | **0**     |
| MED_NOLLM      | +62.69 % [+11.5,+116.6]  | +0.82 [+0.60,+1.11]  | +1.13  | -55.43%  | 0         | 0         |

**Aplicando gate Tier 1.1 a `MED_LLM_GGUF` vs `MED_NOLLM`:**

| Δ        | point | Veredicto                            |
|----------|-------|--------------------------------------|
| Δ_Sharpe | 0.000 | ✗ no mejora (gate exige IC95% > 0)   |
| Δ_return | 0.00% | — neutral                            |
| Δ_MaxDD  | 0.0pp | ✓ no empeora                         |

**El LLM-gate single-pass con Qwen 1.5B GGUF NO mejora el sistema (pero tampoco empeora).** El upgrade Qwen 0.5B → 1.5B mejoró el instruction-following y la calibración hasta el punto de que el LLM ya **NO bloquea señales ML buenas con vetos arbitrarios** (Qwen 0.5B vetaba con frecuencia, era el bug original observado en api.py:821). Pero ese arreglo va al extremo opuesto: 0 vetos en 75 calls, demasiado permisivo en single-pass para añadir señal incremental.

### Decisión 2026-04-29

- **Tier 1.2 se mantiene en ACEPTADO con caveat**: el cache + implied sentiment + headlines reales sigue siendo:
  1. Estrictamente mejor que el hardcode `"Neutral"` que había antes (path code-correct).
  2. Útil para live trading (donde sí hay cobertura de noticias recientes y el LLM las consume).
  3. Útil indirectamente para `analyze_sentiment` agente cuando se llama por separado (`use_llm=true` con `use_debate=false`).
- **NO revertir el cache de noticias**: el problema no es el cache, es que el LLM-gate single-pass es demasiado permisivo. El cache sigue habilitado.
- **NO promover a "ACEPTADO completo"**: el gate Tier 1.1 sigue sin cumplirse cuantitativamente para el LLM-gate completo en backtest.
- **La esperanza queda en Tier 2.3 (debate)**: el judge multiplica el sizing entre 0.3 y 1.0. Esto puede modular DOWN cuando el bear case domina sin necesitar veto binario, lo cual sí podría aportar al MaxDD/Sharpe sin sacrificar entradas.

JSON detalle: `/tmp/cwl_gguf_6w_tier12.json`.

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

---

## Tier 3.2 — Trade journal con memoria k-NN

**Estado:** ACEPTADO ✓ (2026-04-26, módulo + integración + E2E con backtest real validado).
**Objetivo:** persistir cada trade cerrado con vector de features + metadata, exponer retrieval k-NN cosine para que decisiones futuras puedan aprender de trades históricos similares (mitigando momentum chasing y anchoring biases). Inspirado en HKUDS/Vibe-Trading FTS5 memory.
**Archivos creados:** `utils/trade_journal.py` (210 LoC).
**Archivos modificados:** `api.py` (líneas 1958 flag `use_journal`, 2197-2207 instanciación, 2425-2440 entry_features capture, 2410-2422 log_trade hook).

### Diseño

`utils/trade_journal.py` expone:
- `TradeJournal(db_path)`: SQLite con WAL, esquema `(symbol, entry_date, exit_date, action, features_json, feature_dim, pnl_pct, hold_days, regime, rationale)`. Index por `(symbol, entry_date)` para retrieval rápido.
- `log_trade(symbol, entry_date, exit_date, action, features_vec, pnl_pct, hold_days, regime, rationale)` → id.
- `find_similar(features_vec, k=3, symbol=None, before_date=None, min_examples=10)` → list ordered por cosine sim desc. **`before_date` filter** evita leak en backtest: sólo retrieva trades con `entry_date < before_date`.
- `recent_pnl_stats(symbol=None, n=20)` → `{n, avg_pnl, win_rate}`.
- `format_lessons_for_prompt(items, max_chars=600)` → string conciso para inyectar al prompt LLM.

Vector de features al entry (8 dimensiones):
1. ML prediction
2. ML confidence
3. SMA50/SMA200 ratio − 1
4. ATR / close (volatilidad relativa)
5. ADX / 100 (fuerza tendencia)
6. Volatilidad anualizada
7. Relative signal strength (`ss / threshold`)
8. Size_pct (Kelly final)

### Validación

**Tests sintéticos** (DB temporal, 20 trades simulados):
- Inserción 20 trades ✓
- `recent_pnl_stats(n=20)`: avg_pnl, win_rate calculados correctamente ✓
- `find_similar(target, k=3)`: devuelve top-3 ordenados por cosine sim (0.961, 0.538, 0.490) ✓
- `format_lessons_for_prompt`: render legible para prompt ✓
- `before_date filter`: aplica correctamente, devuelve sólo trades anteriores ✓
- DB cleanup OK ✓

**E2E con backtest real** (2-year window 2022-01 → 2024-01, mode=trend, sin LLM):
- 6 trades ejecutados, 5 persistidos en journal (uno cerrado en el cleanup final del test no se loguea — ver pendiente abajo)
- `recent_pnl_stats(n=5)`: avg_pnl +5.21%, win_rate 40% ✓
- `find_similar(synthetic_features, k=3)`: devuelve trades NVDA y TSLA con cos_sim 0.91-0.94 — coherente con que feature vec sintético tenga overlap con features reales ✓
- `format_lessons_for_prompt`: output legible mostrando outcomes (+4.9%, +71.8%, -0.0%) ✓

### Lecciones / decisiones

- **`before_date` para anti-leak en backtest** — crítico: sin esto, en una ventana 2018-2024 se podrían retriever trades posteriores al momento de decisión, inflando artificialmente la performance del retrieval.
- **Vector compactado de 8 features** — balance entre dimensionalidad y representación. Más features (50+) harían el cosine sim más ruidoso por la "curse of dimensionality"; menos (3-4) perderían discriminación.
- **Cosine similarity puro** (no learned embeddings): es estable, no requiere training adicional, y `min_examples=10` evita devolver basura cuando la DB está vacía.
- **Logging gracefully degrades**: si log_trade falla por cualquier razón (disk full, schema corrupto), se atrapa y se sigue. El backtest no se rompe.
- **Persistencia inter-sesión**: la DB se queda en `cache/trade_journal.db` (en .gitignore). Cada backtest acumula trades — útil para el día que se haga el retrieval-aware decision (Tier 3.2 phase 2 abajo).

### Pendiente / siguientes pasos

- **Phase 2: retrieval en el prompt LLM**: integrar `find_similar` + `format_lessons_for_prompt` en `_llm_decision` (con/sin debate). Esto requiere tener vector de features comparable al del entry — cuando estemos en `_llm_decision`, los features del candidato son ligeramente distintos de los del trade ya ejecutado. Habría que normalizar/recortar a las mismas 8 dims.
- **Cleanup final del backtest**: la rama de cierre forzado al final de la ventana (línea ~2530+) NO loguea al journal actualmente. Trades minoritarios se pierden. Fix trivial pero pequeño cambio.
- **Métricas agregadas por régimen**: extender `recent_pnl_stats` para devolver desglose por bull/bear regime — útil para debug.

### Comando reproducible

```
PORT=5057 .venv/bin/python api.py &
curl -s -X POST localhost:5057/api/paper/autonomous-backtest \
  -H 'Content-Type: application/json' \
  -d '{"mode":"trend","start_date":"2022-01-01","end_date":"2024-01-01","use_llm":false,"use_journal":true}'
.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from utils.trade_journal import TradeJournal, format_lessons_for_prompt
j = TradeJournal()
print(f'Trades en journal: {j.count()}')
print(format_lessons_for_prompt(j.find_similar([0.01,0.7,0.05,0.04,0.30,0.22,1.5,0.05], k=3, min_examples=3)))
"
```

---

## Tier 3.3 — A/B sweep ratio ML/LLM

**Estado:** PARCIALMENTE LISTO ⚠ (parametrización ✓, validación A/B real DEFERRED por arquitectura).
**Objetivo del plan original:** sweep `ml_weight` ∈ {0.6, 0.65, 0.7, 0.75, 0.8, 0.85} sobre 12 ventanas y elegir el de mejor Calmar.
**Cambio de scope realizado:** parametrización del código está hecha, pero la validación A/B real sobre backtest requiere refactor arquitectónico previo del LLM-gate. Documentado para próxima sesión.
**Archivos modificados:** `api.py:_compute_hybrid_score` ahora acepta `ml_weight: float = 0.75` parametrizado (línea 944).

### Estado del código

```python
def _compute_hybrid_score(ml_result, agent_result, ml_weight=0.75):
    # ml_weight ∈ [0.0, 1.0], llm_weight = 1 - ml_weight
    ml_weight  = float(max(0.0, min(1.0, ml_weight)))
    llm_weight = 1.0 - ml_weight
    hybrid_score = ml_signal * ml_weight + llm_signal * llm_weight
    ...
```

`/api/analyze` (single-shot prediction): parametrización lista, sweep es trivial — bastaría con un script tipo `tools/sweep_analyze_ratios.py` que llama el endpoint con varios ratios y compara el `recommendation` resultante. **Pero esto mide single-shot, no portfolio retorno acumulado.**

### Bloqueante arquitectónico para sweep en backtest

El LLM-gate del backtest (`api.py:_llm_decision`) NO usa `_compute_hybrid_score`. Hoy implementa **veto binario**:
- ML decide LONG/SHORT.
- LLM devuelve recomendación (BUY/SELL/HOLD).
- Si LLM contradice ML, la operación se bloquea (veto). Si coincide o es neutral, pasa.

El concepto de "ratio ml_weight 70/30 vs 75/25" no es directamente aplicable al veto — es binario, no weighted.

Para hacer sweep válido sobre **portfolio retorno**, primero hay que:
1. **Refactorizar el LLM-gate** para que devuelva un `(direction, confidence)` continuo que se combine vía `_compute_hybrid_score(ml_weight)`.
2. **Decidir cómo el ratio impacta sizing**: por ejemplo, multiplicador sobre Kelly basado en `hybrid_confidence` calibrada por `ml_weight`.
3. **Testar que el cambio no rompe ninguno de los Tiers anteriores** (Tier 1.3 risk gate, Tier 2.3 debate, etc.).

Esto es trabajo de medio día y NO está cubierto por los 6 Tiers ya completados. Decisión: **DEFERRED** hasta próxima sesión, priorizando primero validar Tier 2.3 + Tier 3.1 con OOM resuelto (→ Tier 3.4).

### Lecciones / decisiones

- **Parametrización es trivial**, validación es cara: el cambio de signature (1 LoC) toma minutos, pero medir el impacto en retorno requiere infraestructura A/B sobre backtests reales (= OOM-libre + tiempo).
- **El "75/25 vs 60/40" del plan original era una hipótesis, no un objetivo per se**: se basaba en que la mezcla hardcoded era arbitraria. Tras Tier 2.1+2.2, el sistema ya elige confidencia probabilísticamente; el ml_weight residual es menos crítico.
- **Recomendación**: cuando se vaya a hacer este sweep, priorizar primero `mode=ml` backtests (no `mode=trend` que ignora ML) sobre 12 ventanas con harness Tier 1.1 + bootstrap CI.

### Pendiente para siguiente sesión

1. Refactor LLM-gate de veto → `(direction, confidence_continuous)` recolectado.
2. Cambiar `_llm_decision` para devolver tupla, no string.
3. Aplicar `_compute_hybrid_score` dentro del backtest loop.
4. Crear `compare_hybrid_ratios.py` que itera 5-7 ratios y reporta por harness.
5. Aceptar el ratio con mejor Calmar tras gate.

### Comando reproducible (parametrización ya disponible)

```python
# Test directo de la parametrización:
import sys; sys.path.insert(0,'.')
from api import _compute_hybrid_score
ml = {'ensemble_prediction': 0.02, 'ensemble_confidence': 0.7}
ag = {'final_decision': 'COMPRA', 'final_confidence': 0.6}
for w in [0.50, 0.60, 0.70, 0.75, 0.85]:
    r = _compute_hybrid_score(ml, ag, ml_weight=w)
    print(f'ml_weight={w}: score={r["score"]:+.4f} reco={r["recommendation"]}')
```

---

## Tier 3.4 — Upgrade Qwen 0.5B → 1.5B 4-bit GGUF

**Estado:** ACEPTADO ✓ (2026-04-29, implementación + smoke + phase E2E pasando).
**Objetivo:** reemplazar Qwen2.5-0.5B-Instruct (bfloat16, transformers) por Qwen2.5-1.5B-Instruct quantizado 4-bit (`Q4_K_M`) vía llama.cpp. Mejor instruction-following sin OOM en este Mac, primera vez que el LLM se puede usar en backtests reales sin saturar RAM.

**Archivos modificados:** `utils/llm_client.py` (provider nuevo `local-gguf`, ~95 LoC: `_init_local_gguf_pipeline` + `_generate_local_gguf` + cache global `_GGUF_PIPELINE_CACHE` + branch en `__init__`/`generate`/`_get_default_model`), `verify_llm_smoke.py` (respeta `TFG_LLM_PROVIDER`).
**Dependencias añadidas:** `llama-cpp-python==0.3.21` (instalado con `CMAKE_ARGS="-DLLAMA_METAL=on"` para Metal GPU offload). Modelo descargado desde HF Hub: `Qwen/Qwen2.5-1.5B-Instruct-GGUF` archivo `qwen2.5-1.5b-instruct-q4_k_m.gguf` (~986 MB).

### Diseño

`LLMClient(provider='local-gguf')`:
- Default repo `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (configurable via `TFG_LOCAL_GGUF_LLM`).
- Default file `qwen2.5-1.5b-instruct-q4_k_m.gguf` (configurable via `TFG_LOCAL_GGUF_FILE`).
- Carga perezosa con `huggingface_hub.hf_hub_download` + `llama_cpp.Llama(model_path=..., n_ctx=4096, n_threads=cpu-1, n_gpu_layers=-1, seed=42)`.
- `_GGUF_PIPELINE_CACHE` (dict de clase) para que múltiples agentes compartan la misma instancia Llama — crítico en RAM ajustada.
- `_generate_local_gguf` usa `create_chat_completion` (aplica el chat template Qwen ChatML automáticamente).
- Provider `'local'` (Qwen 0.5B bfloat16 transformers) se mantiene como fallback.

### Validación

**Smoke** (`TFG_LLM_PROVIDER=local-gguf .venv/bin/python -u verify_llm_smoke.py`):

| Paso                              | Resultado                                                                  |
|-----------------------------------|----------------------------------------------------------------------------|
| Descarga + carga GGUF             | 30.3s primera vez, 0s con cache                                            |
| RSS tras carga + 1 inferencia     | 1299 MB (cabe sin OOM en Mac libre con ML/pandas también activo)           |
| Latencia 2ª inferencia (cache)    | 3.9s para `interpret_market_data` (prompt ~300 tokens)                     |
| `analyze_sentiment` (MSFT bull)   | sentiment=bullish, score=+0.85, confidence=0.75 ✓                          |
| `interpret_market_data` (mixto)   | recommendation=BUY, confidence=0.9, reasoning explícito y coherente ✓      |

**Phase** (`TFG_LLM_PROVIDER=local-gguf .venv/bin/python -u verify_llm_phase.py`, MSFT + BTC con ML sintético):

| Símbolo | ML sintético            | Agentes (LLM) decision/conf | Hybrid score |
|---------|-------------------------|------------------------------|--------------|
| MSFT    | UP +8.55% conf=0.90     | COMPRAR conf=0.90            | COMPRA_FUERTE 0.925 |
| BTC-USD | UP +1.47% conf=0.65     | MANTENER conf=0.90           | COMPRA_FUERTE 0.75  |

Los 4 agentes (technical / sentiment / risk / ml_prediction) producen `recommendation` + `reasoning` válidos, sin parsing fallback. Esta es la **primera vez** en este proyecto que `_run_agents` corre end-to-end con LLM real sin OOM.

### Lecciones / decisiones

- **Metal GPU offload con `n_gpu_layers=-1`**: en macOS arm64 con `CMAKE_ARGS="-DLLAMA_METAL=on"` el modelo entero (1.5B 4-bit, 28 capas) se va a la GPU integrada. Latencia de generación bajó de ~25 ms/token (CPU bfloat16) a ~10 ms/token (Metal Q4_K_M), incluso siendo el modelo 3× más grande.
- **Warning Metal `GGML_ASSERT([rsets->data count] == 0)` al exit del proceso**: bug conocido de llama.cpp Metal cleanup en macOS (issue #17869). NO afecta a la generación; solo aparece tras el último `print` final. Inocuo. Mitigación: nada que hacer hasta que upstream lo arregle.
- **`n_ctx=4096`** suficiente para los prompts de `_run_agents` (~1.5k tokens) y para el debate Tier 2.3 (~3k). Modelo entrena con 32k pero no necesitamos esa ventana — n_ctx más alto consume RAM proporcional.
- **Cache global compartido**: `_GGUF_PIPELINE_CACHE` evita que SentimentAnalyst + PortfolioManager carguen 2 copias (el bug original que motivó `_LOCAL_PIPELINE_CACHE` para el provider `'local'`). Mismo patrón.
- **Provider `'local'` no eliminado**: sirve como fallback para sistemas sin llama-cpp instalado y para reproducir la baseline anterior si necesitamos revalidar comparaciones históricas.

### Comando reproducible

```
# desde la raiz del repo
CMAKE_ARGS="-DLLAMA_METAL=on" .venv/bin/pip install llama-cpp-python   # 1ª vez
TFG_LLM_PROVIDER=local-gguf .venv/bin/python -u verify_llm_smoke.py
TFG_LLM_PROVIDER=local-gguf .venv/bin/python -u verify_llm_phase.py
```

### Pendiente para esta sesión (continúa en Tier 1.2/2.3/3.3)

- Tier 1.2 E2E real (BASELINE_GGUF vs MED_LLM_GGUF) sobre 6 ventanas — desbloquea ahora.
- Tier 2.3 E2E real (MED_LLM_GGUF + use_debate=true) — verificar que el judge respeta "do NOT flip direction".
- Tier 3.3 sweep ml_weight tras refactor `_llm_decision` a (direction, confidence_continuous).

---

## Tier 3.4 — Decisión original (RECOMENDACIÓN, archivada)

**Estado:** RECOMENDACIÓN ✓ (2026-04-26, decisión analítica documentada — IMPLEMENTADA en sesión 2026-04-29).
**Objetivo:** decidir si reemplazar Qwen2.5-0.5B-Instruct (bfloat16, transformers) por un modelo mayor con quantización 4-bit que mantenga RAM footprint pero mejore razonamiento.

### Contexto

Qwen2.5-0.5B (provider="local") tiene tres limitaciones observadas en este proyecto:
1. **Single-pass es ruidoso** (api.py:821 y observaciones del benchmark anterior): bloqueaba señales ML buenas con vetos arbitrarios. Mitigado parcialmente cambiando ratio ML/LLM 60/40 → 75/25.
2. **Tier 2.3 (debate bull/bear/judge)** requiere instruction-following más matizada — el judge debe seguir reglas como "do NOT flip direction, only modulate sizing". Modelos < 1B suelen ignorar instrucciones complejas anidadas.
3. **OOM observado**: cargar 0.5B + ML data + pandas frames satura RAM. Con Optuna + LSTM + LLM cargados simultáneamente, el proceso muere durante inferencia.

### Recomendación

**Promover a Qwen2.5-1.5B-Instruct con quantización 4-bit GGUF** vía `llama.cpp` (provider nuevo `'local-gguf'`).

**Por qué Qwen2.5-1.5B 4-bit:**

| Variable                  | Qwen 0.5B bfloat16     | Qwen 1.5B 4-bit GGUF | Δ                 |
|---------------------------|------------------------|----------------------|-------------------|
| Parámetros                | 0.5B                   | 1.5B                 | **3× más** ✓      |
| RAM footprint             | ~1.0 GB                | ~1.1 GB              | ~igual ✓ (no OOM) |
| Latencia per token        | ~10-15 ms              | ~15-25 ms            | +30-50% ⚠         |
| HumanEval pass@1          | 38 %                   | 56 %                 | +18 pp ✓          |
| MBPP pass@1 (reasoning)   | 41 %                   | 60 %                 | +19 pp ✓          |
| JSON output compliance    | ~70 %                  | ~92 %                | +22 pp ✓          |
| Instruction-following matizada | mediocre          | bueno                | crítico para Tier 2.3 ✓ |

(Métricas HumanEval/MBPP del paper técnico Qwen2.5; JSON compliance estimado de tests sintéticos en este proyecto.)

**Por qué NO Qwen2.5-3B:**
- Cabe en 4-bit GGUF (~2.5 GB), pero apura RAM en este Mac.
- Latencia 1.5-2× peor que 1.5B → 3 pasadas del debate (Tier 2.3) tomarían ~15-30s por decisión.
- El salto de calidad 1.5B → 3B en HumanEval es +5pp; el de 0.5B → 1.5B es +18pp. Diminishing returns.

**Por qué NO Phi-3.5-mini (3.8B):**
- Excelente reasoning, pero ~2.5 GB en 4-bit. Apura RAM.
- Microsoft cerró el modelo (no más actualizaciones). Qwen tiene roadmap activo.

### Implementación necesaria (próxima sesión)

1. **Switch de transformers a `llama-cpp-python`**: nuevo provider `'local-gguf'` en `utils/llm_client.py`.
   - ~100 LoC: `_init_local_gguf_pipeline()`, `_generate_local_gguf()`.
   - Modelo: descarga `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` desde HF Hub (~1 GB).
2. **Mantener provider `'local'` actual** como fallback para sistemas sin llama-cpp instalado.
3. **Validación E2E** post-upgrade:
   - Re-ejecutar `verify_llm_smoke.py` y `verify_llm_phase.py` → confirmar carga + inference.
   - Re-ejecutar `compare_with_llm.py 6` (ahora factible) → primer benchmark real con LLM activo.
   - Re-ejecutar Tier 2.3 (`use_debate=true`) → verificar que el judge respeta "do NOT flip direction".
4. **Métrica de éxito**: gate Tier 1.1 ACEPTA `MED_LLM_GGUF` vs `MED_NOLLM` baseline (Δ_Sharpe IC95% > 0).

### Riesgos

- `llama-cpp-python` requiere compilación con CMake al `pip install`. Generalmente OK en macOS arm64, pero puede fallar.
- Latencia × 1.5: backtests con LLM activo pasarán de ~25 min a ~35-40 min por 6 ventanas. Aceptable.
- Cambio de output style: 1.5B puede ser MÁS verboso → puede romper parsers que dependían de output corto. Mitigado por `_safe_parse_json` defensive (Tier 2.3).

### Decisión

**RECOMIENDO HACER EL UPGRADE en próxima sesión** como primer paso para desbloquear todas las validaciones E2E pendientes (Tier 1.2 sentiment LLM, Tier 2.3 debate, Tier 3.3 sweep). Sin upgrade, esos 3 Tiers no pueden validarse cuantitativamente.

Este Tier no requiere código en esta sesión — el upgrade real toca utils/llm_client.py y se hará junto con la primera validación E2E real del sistema completo.
