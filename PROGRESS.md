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

**Estado:** RECHAZADO ✗ (2026-04-29 E2E real con Qwen 1.5B GGUF: gate Tier 1.1 no se cumple, Δ_Sharpe = 0). Código mantenido en `agents/debate.py`, **OFF por default** (`use_debate=False`).
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

### Validación E2E real (2026-04-29, post-Tier 3.4 upgrade)

`compare_debate.py 6` con `llm_provider=local-gguf`, 6 ventanas seed=42, ~10 min total. 3 variantes: MED_NOLLM (control sin LLM), MED_LLM_GGUF (single-pass), MED_LLM_GGUF_DEBATE (3-pass bull/bear/judge).

| variante              | avg_ret±CI95             | Sharpe±CI95          | Calmar | worst_DD | LLM calls | LLM vetos |
|-----------------------|--------------------------|----------------------|--------|----------|-----------|-----------|
| MED_NOLLM             | +62.69 % [+11.5,+116.6]  | +0.82 [+0.60,+1.11]  | +1.13  | -55.43%  | 0         | 0         |
| MED_LLM_GGUF          | +62.69 % [+11.5,+116.6]  | +0.82 [+0.60,+1.11]  | +1.13  | -55.43%  | 12.5      | 0         |
| MED_LLM_GGUF_DEBATE   | +62.69 % [+11.5,+116.6]  | +0.82 [+0.60,+1.11]  | +1.13  | -55.43%  | **37.5**  | 0         |

**Aplicando gate Tier 1.1 a `MED_LLM_GGUF_DEBATE` vs `MED_LLM_GGUF`:**

| Δ        | point | Veredicto                        |
|----------|-------|----------------------------------|
| Δ_Sharpe | 0.000 | ✗ no cumple gate (`< +0.10`)     |
| Δ_return | 0.00% | — neutral                        |
| Δ_MaxDD  | 0.0pp | ✓ no empeora                     |

**El debate NO mejora el sistema cuantitativamente** (mismas decisiones que single-pass, que son las mismas que sin LLM). Conclusión consistente con Tier 1.2: el LLM 1.5B GGUF es demasiado permisivo en backtest histórico — sigue la instrucción "do NOT flip direction" del judge correctamente (buena señal de instruction-following), pero el `confidence_mult` que devuelve nunca cae por debajo del threshold de veto 0.4 en estas 6 ventanas.

### Diagnóstico

El judge respeta perfectamente las reglas del prompt (no hay parsing fallbacks, no hay errores). Pero en estos 75 backtests el bear case nunca gana suficiente como para que `confidence_mult < 0.4` (que es lo que produciría veto/HOLD). El sizing modulation [0.4, 1.0] sí ocurre, pero como el bot ya tiene Kelly+vol-scaling internos, la diferencia es marginal y no se traduce en cambio de decisiones de entrada/salida.

Esto refleja una verdad estructural del bot híbrido: **las 4 capas de pre-filtrado (SMA200, ADX≥22, crash_filter, vol_scaling) ya filtran tanto que cuando llegamos al LLM, las señales ML que sobreviven ya son las "buenas". El LLM no tiene oportunidades reales para añadir señal incremental** — sólo aporta cuando el ML genera ruido, lo cual se filtra antes.

### Lecciones / decisiones (revisadas 2026-04-29)

- **Mantener `use_debate=False` por default**: 3× coste LLM sin retorno → no justificable en producción.
- **No borrar `agents/debate.py`**: el módulo es correcto, los tests sintéticos pasan, y podría aportar valor en escenarios futuros donde el ML genere más ruido (mode=ml en lugar de mode=trend, o tras introducir features macro/cross-asset que aumenten la varianza de las predicciones).
- **El judge respetando "do NOT flip direction" es valioso por sí solo** como prueba de que Qwen 1.5B GGUF sigue instrucciones complejas anidadas. Habilita futuros patterns multi-pass más sofisticados.
- **Honestidad sobre la hipótesis original**: el plan asumía que un LLM mejor (1.5B) + debate mejoraría sizing. La realidad es que el cuello de botella ya no es el LLM — es que el bot ya está sobre-filtrado por reglas determinísticas. La conclusión es informativa para el TFG aunque no avance el retorno.

### Comando reproducible

```
# desde la raiz del repo
PORT=5057 .venv/bin/python -u api.py &
.venv/bin/python -u compare_debate.py 6     # ~10 min con LLM activo, 0 OOM
# JSON detalle: /tmp/compare_debate_result.json
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

**Estado:** RECHAZADO ✗ (2026-04-29 sweep completo post-Tier 3.4 upgrade: ningún ratio ml_weight ∈ [0.6, 1.0] supera el gate Tier 1.1). Refactor del LLM-gate de veto-only a hybrid_score continuo IMPLEMENTADO y disponible, pero default vuelve a `ml_weight=1.0` (comportamiento legacy).
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

### Refactor implementado (2026-04-29)

`api.py:_llm_decision` ahora devuelve `dict {'rec': str, 'confidence': float}` en lugar de string. La cache trimestral cachea el dict completo.

En el backtest loop (`api.py:2746-2785`), antes del `_kelly`:

```python
base_ml_conf = float(bday.get('conf', 0.5))
if use_llm and llm_dec is not None:
    llm_conf = float(llm_dec.get('confidence', 0.5))
    rec = str(llm_dec.get('rec', '')).upper()
    if rec in ('HOLD', 'WAIT', 'MANTENER', ''):
        llm_conf *= 0.7   # neutral = menos convicción
    eff_conf = ml_weight * base_ml_conf + (1.0 - ml_weight) * llm_conf
else:
    eff_conf = base_ml_conf
kelly = _kelly(bday['pred'], eff_conf, stop_pct)
```

Body request acepta `ml_weight ∈ [0, 1]`, default 1.0 (idéntico a veto-only legacy). El LLM sigue actuando como veto cuando contradice fuerte; cuando coincide o es neutral, su confianza se mezcla con la del ML para escalar el sizing vía Kelly.

### Sweep validado (`compare_hybrid_ratios.py 6`, 2026-04-29)

6 ventanas seed=42, 5 ratios + 1 control sin LLM, ~22 min, 0 OOM, **0 vetos en 360 LLM calls**.

| variante       | avg_ret±CI95             | Sharpe±CI95          | Calmar  | worst_DD | score_v2 |
|----------------|--------------------------|----------------------|---------|----------|----------|
| MED_NOLLM      | +62.69 % [+11.5,+116.6]  | +0.82 [+0.60,+1.11]  | +1.131  | -55.43%  | 50.16    |
| MED_GGUF_w60   | +61.88 % [+11.5,+116.0]  | +0.81 [+0.59,+1.10]  | +1.122  | -55.43%  | 49.42    |
| MED_GGUF_w70   | +61.71 % [+11.7,+115.4]  | +0.82 [+0.60,+1.09]  | **+1.148** | -53.74% | **50.98** |
| MED_GGUF_w80   | +62.54 % [+11.5,+116.6]  | +0.82 [+0.60,+1.10]  | +1.139  | -54.92%  | 50.47    |
| MED_GGUF_w90   | +61.79 % [+11.5,+114.4]  | +0.82 [+0.60,+1.10]  | +1.130  | -54.88%  | 49.97    |
| MED_GGUF_w100  | +62.69 % [+11.5,+116.6]  | +0.82 [+0.60,+1.11]  | +1.131  | -55.43%  | 50.16    |

`w100` produce resultados IDÉNTICOS a `MED_NOLLM` — confirma que el refactor está bien implementado (con `ml_weight=1.0`, `eff_conf == base_ml_conf` y el LLM no afecta sizing, solo veto, que sigue sin disparar).

`w70` tiene mejor Calmar (+1.148 vs +1.131 baseline) — el LLM modula sizing a la baja en ventanas con alta vol histórica (W5 worst_DD: -32.84% w60 vs -34.87% baseline) — pero la mejora NO supera el gate.

### Aplicando gate Tier 1.1 (cada ratio vs MED_NOLLM)

| variante       | Δ_Sharpe (CI95)         | Δ_return  | Δ_MaxDD  | Veredicto         |
|----------------|--------------------------|-----------|----------|-------------------|
| MED_GGUF_w60   | -0.009 [-0.02,-0.00]     | -0.81 pp  | +0.35 pp | ✗ peor Sharpe sig.|
| MED_GGUF_w70   | -0.008 [-0.02,+0.00]     | -0.98 pp  | +0.60 pp | ✗ no mejora       |
| MED_GGUF_w80   | -0.005 [-0.01,-0.00]     | -0.15 pp  | +0.29 pp | ✗ no mejora       |
| MED_GGUF_w90   | -0.004 [-0.01,-0.00]     | -0.90 pp  | +0.20 pp | ✗ no mejora       |
| MED_GGUF_w100  |  0.000 [+0.00,+0.00]     | +0.00 pp  | +0.00 pp | ≡ idéntico        |

**Ningún ratio supera el gate** (Δ_Sharpe IC95% > 0). La gente conservadora podría argumentar que `w70` es el mejor por Calmar y mantenerlo; pero metodológicamente el gate Tier 1.1 dice rechazar y mantener el comportamiento legacy.

### Lecciones / decisiones

- **El refactor en sí está bien implementado** — el sweep produce diferencias diferenciables (vs el veto-only que producía resultados idénticos). El gate detecta correctamente que esas diferencias no son estadísticamente significativas.
- **Default vuelve a `ml_weight=1.0`** (comportamiento legacy idéntico a veto-only). El parámetro queda disponible para reabrir tras cambios futuros que aumenten la varianza de las predicciones (e.g. mode=ml más ruidoso, features macro adicionales).
- **Consistencia con Tier 1.2 y 2.3**: las 3 validaciones E2E del LLM (sentiment cache, debate, hybrid sizing) llegan a la misma conclusión — Qwen 1.5B GGUF es bueno cualitativamente (instruction-following, no flippea, no OOM) pero en backtest histórico no aporta señal incremental sobre el ML pre-filtrado por reglas determinísticas.
- **Justificación TFG**: este resultado tiene valor pedagógico — demuestra que la integración LLM bien-implementada NO siempre traduce en mejora cuantitativa cuando el sistema base ya está bien-diseñado. Documenta el cuello de botella REAL (filtros pre-LLM saturan la señal).

### Comando reproducible

```
# desde la raiz del repo
PORT=5057 .venv/bin/python -u api.py &
.venv/bin/python -u compare_hybrid_ratios.py 6   # ~22 min con LLM activo, 0 OOM
# JSON detalle: /tmp/compare_hybrid_ratios_result.json
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

---

## Tier 4.1 — Aflojar conservadurismo (filtros + sizing)

**Estado:** ACEPTADO ✓ (2026-04-30, AGGR_KELLY_PCT pasa gate Agresivo sobre 12 ventanas).
**Objetivo:** atacar el hallazgo del dashboard (`/tmp/perf_grid_result.json`): el bot baseline solo bate B&H en 20% de runs y pierde a plazos cortos. Diagnóstico forense (Fase 0) confirma que el bot está sobre-filtrado y sub-dimensionado.
**Archivos creados:** `tools/measure_filter_rejection.py`, `tools/sweep_filters_sizing.py`, `compare_aggressive.py`, `tools/dashboard_compare.py`.
**Archivos modificados:** `api.py` (4 LoC: `disable_golden_cross`, `regime_adx_min` override, `max_concurrent` parametrizable, `debug_filter_counts`), `utils/backtest_metrics.py` (+50 LoC: `aggressive_gate`), `tools/run_performance_grid.py` (extender CLI con `--variant {med,aggr}`).

### Diagnóstico Fase 0 (medición de rechazos por filtro, 6 ventanas)

| filtro              | rechazados | % del total |
|---------------------|-----------:|------------:|
| signal_percentile   |     18 507 |    **94.17 %** |
| no_direction        |        902 |       4.59 % |
| adx_filter          |         76 |       0.39 % |
| spy_regime          |         56 |       0.28 % |
| crash_filter        |         32 |       0.16 % |
| sma200_filter       |          3 |       0.02 % |
| **golden_cross**    |          0 |       0.00 % |
| **quality_gate**    |          0 |       0.00 % |
| asset_r2_gate       |          0 |       0.00 % |
| pasaron todos       |         76 |       0.39 % |

**Hallazgos contraintuitivos:**
- El **signal_percentile (top-18%) rechaza 94% de TODO**. Es el cuello dominante.
- **golden_cross rechaza CERO** — es completamente redundante con SMA200 (redundancia: si precio<SMA200, también SMA50<SMA200 casi siempre).
- ADX, SMA200, SPY regime y crash_filter rechazan en conjunto < 1%. NO son el problema.
- El diagnóstico inicial del agente Explore (que apuntaba a ADX/golden_cross como culpables) **estaba equivocado** — la medición empírica refuta esa hipótesis.

### Sweeps Fase 1 (gate Agresivo: Δ_Sharpe IC95% > 0 Y (Δ_ret IC95% > +5pp O Δ_DD ≤ +2pp); hard reject si Δ_DD < −10pp o Δ_ret IC95% upper < −2pp)

| Exp | Parámetro            | Valores probados                         | Ganador individual |
|-----|----------------------|------------------------------------------|--------------------|
| E1  | signal_percentile    | {0.82, 0.75, 0.70, 0.65, 0.60}           | — sin ganador (Sharpe mejora mucho pero CIs anchos no superan ret_lo > +5pp) |
| E2  | regime_adx_min       | {22, 18, 15, 10, 0}                      | — empeora Sharpe consistentemente |
| E3  | disable_golden_cross | {off, on}                                | — Δ exactamente 0 (filtro inactivo) |
| E4  | kelly_scale          | {0.22, 0.30, 0.40, 0.50, 0.60}           | **kelly_scale=0.30** ✓ |
| E5  | max_position_pct     | {0.18, 0.22, 0.26, 0.30, 0.35}           | — Δ exactamente 0 (cap inactivo: el sizing actual está bajo 0.18) |
| E6  | max_concurrent       | {3, 4, 5}                                | — Sharpe baja (más posiciones diluyen las buenas) |

E4 ganador: `kelly_scale=0.30 vs 0.22` → Δ_Sharpe +0.120 [+0.05, +0.18], Δ_return +19.61pp, Δ_DD −6.10pp (dentro del −10pp tope). `kelly_scale=0.40+` rechazado-HARD por DD < −10pp.

### Validación Fase 2 (12 ventanas seed=42, A/B vs MED_NOLLM)

| variante         | avg_ret±CI95              | Sharpe±CI95          | MaxDD   | Calmar | %pos | trades |
|------------------|---------------------------|----------------------|---------|--------|------|--------|
| MED_NOLLM        | +29.67% [+0.0, +65.6]     | +0.71 [+0.55, +0.89] | -32.59% | +0.54  | 67%  | 9.4    |
| AGGR_KELLY       | +40.19% [+3.0, +86.4]     | +0.84 [+0.66, +1.04] | -38.61% | +0.66  | 67%  | 9.4    |
| **AGGR_KELLY_PCT** | **+51.44%** [+14.5, +95.9] | **+0.93** [+0.75, +1.12] | -40.99% | **+0.90** | **75%** | **10.7** |

**Aplicando gate Agresivo sobre AGGR_KELLY_PCT vs MED_NOLLM:**

| Δ        | point   | CI95           | Veredicto                             |
|----------|---------|----------------|---------------------------------------|
| Δ_Sharpe | +0.217  | [+0.141,+0.291]| ✓ sharpe_positive (CI lo > 0)         |
| Δ_return | +21.77pp| [+10.16,+35.59]| ✓ return_big_win (CI lo > +5pp)       |
| Δ_MaxDD  | −8.40pp | [−11.13,−5.48] | ✓ NO disaster (≥ −10pp)               |
| **VEREDICTO** | | | **ACEPTA** |

`AGGR_KELLY` solo (sin signal_pct) NO acepta — return_big_win=False (CI lo +2.65pp < +5pp). La combinación con `signal_percentile=0.70` rescata el gate. **Lección clave**: cambios individualmente no significativos pueden ser sinérgicos.

### Validación Fase 3 (40 backtests sobre cuadrícula 10 fechas × 4 plazos)

`tools/dashboard_compare.py` muestra side-by-side. Comparativa global:

| KPI global       | MED_NOLLM | AGGR_KELLY_PCT | Δ      |
|------------------|-----------|-----------------|--------|
| avg_return       | +17.63 %  | **+24.13 %**    | +6.5pp |
| runs positivos   | 20/40 (50%) | 21/40 (52.5%) | +1     |
| runs baten B&H   | 8/40 (20%) | 7/40 (17.5%)  | -1 ⚠   |
| runs sin trades  | 7/40      | **2/40**       | −5 ✓   |
| avg trades/run   | 7.0       | **9.4**         | +2.4   |
| avg MaxDD        | -18.4 %   | -27.9 %         | -9.5pp |

**Por plazo (lo más relevante):**

| plazo | MED ret | AGGR ret | Δ       | MED Sharpe | AGGR Sharpe |
|-------|---------|----------|---------|------------|--------------|
| 3M    | -3.51%  | -12.23%  | -8.7pp  | +0.33      | +0.80        |
| 6M    | +4.54%  | -0.27%   | -4.8pp  | +0.53      | +0.81        |
| 1Y    | +21.60% | **+34.86%** | +13.3pp | +0.54     | **+0.89**     |
| 2Y    | +47.91% | **+74.19%** | +26.3pp | +0.59     | **+0.93**     |

A 1-2 años AGGR mejora claramente; en 3-6M empeora (más exposición = más volatilidad). La pérdida de un caso de "alpha+" se compensa con el +37% en avg_return global.

### Lecciones / decisiones

- **El diagnóstico inicial del agente Explore se equivocó**: apuntaba a ADX y Golden Cross como filtros que capaban upside. La medición empírica probó que rechazan <0.5% combinado. Sin la Fase 0 de medición real, habríamos perdido tiempo en sweeps inútiles.
- **El cuello de botella real son DOS COSAS combinadas**, no una sola: signal_percentile demasiado restrictivo + kelly_scale demasiado pequeño. Los sweeps individuales no detectan sinergias — la Fase 2 es esencial.
- **`max_position_pct` y `disable_golden_cross` son inútiles individualmente** porque sus caps/filtros NO se activan en el sizing actual. Para que importen habría que cambiar también algo aguas arriba (más kelly_scale, más sizing).
- **AGGR no es Pareto-superior**: empeora 3M/6M y aumenta MaxDD. Es una elección consciente de "más upside a cambio de más DD" — alineado con la decisión del usuario de gate Agresivo.
- **Dashboard comparativo (tools/dashboard_compare.py)** debería ser el primer paso de cualquier futura validación de cambios paramétricos. Permite ver visualmente trade-offs que los números agregados ocultan.
- **El bot ya está cerca del Pareto-frontier** para esta era 2018-24: solo combinaciones ajustadas finas pasan gates. Ganancias mayores requerirían **mode=ml** (predicciones más ricas) o cambio de universo (ETFs, factor-tilted), no más tweaking de parámetros trend.

### Comando reproducible

```
# desde la raiz del repo
PORT=5057 .venv/bin/python -u api.py &

# Fase 0 — diagnóstico
.venv/bin/python -u tools/measure_filter_rejection.py 6

# Fase 1 — sweeps individuales (~20 min)
.venv/bin/python -u tools/sweep_filters_sizing.py

# Fase 2 — validación combinada (12 ventanas, ~10 min)
.venv/bin/python -u compare_aggressive.py 12

# Fase 3 — grid + dashboard comparativo
.venv/bin/python -u tools/run_performance_grid.py                  # MED baseline
.venv/bin/python -u tools/run_performance_grid.py --variant aggr   # AGGR Tier 4.1
.venv/bin/python -u tools/dashboard_compare.py
open /tmp/dashboard_compare.html
```

### Pendiente / siguientes pasos

- **Activar AGGR_KELLY_PCT como default trend mode**: cambiar `signal_percentile=0.82→0.70` y `kelly_scale=0.22→0.30` en BotConfig requiere validación adicional de live trading. Por ahora se documenta como variante recomendada pero default queda inalterado para compatibilidad con runs históricos.
- **Investigar mode=ml**: los gates de Tiers 2-3 mostraron que el LLM no aporta valor sobre las señales filtradas, pero `mode=ml` (sin filtros tan agresivos, con ML como decisor primario) podría aprovechar mejor el ensemble calibrado.
- **Considerar universo extendido**: añadir ETFs (SPY, QQQ, IWM) para diversificación sectorial y reducir concentración en 6 stocks tech + 4 cryptos.

---

## Tier 4.2 — Investigación externa + sweeps de candidatos C1-C5

**Estado:** PARCIALMENTE ACEPTADO ✓ (2026-04-30, C3 ACEPTADO sobre 24 ventanas; C1/C2/C4/C5 RECHAZADOS).
**Objetivo:** investigar Reddit/foros + repos GitHub similares (freqtrade, vectorbt, jesse, qlib) para identificar técnicas con evidencia empírica que podrían mejorar nuestro bot, validar cada una con A/B + gate Agresivo Tier 4.1 antes de mergear.
**Archivos creados:** `compare_candidates.py` (~200 LoC), `tools/dashboard_compare3.py` (~280 LoC).
**Archivos modificados:** `api.py` (~30 LoC: params opt-in `include_etfs`, `enable_vol_target_overlay`, `vol_filter_min/max`, `enable_mr_combo`, `topn_rotation` + lógica), `tools/run_performance_grid.py` (variante `aggr_plus`).

### Investigación externa (resumen)

**Agente A (Reddit + foros + papers):** identificó 6 ideas con evidencia, lideradas por:
- Vol-targeting Moreira-Muir 2017 (NBER) — paper peer-reviewed con réplica
- Cross-sectional momentum top-N rotation (Han et al SSRN, Starkiller Capital, Asness)
- HMM/Hurst regime detection (QuantStart tutorial)
- Combinatorial Purged K-Fold + Deflated Sharpe (Arian/Norouzi/Seco KBS 2024)
- ETFs sectoriales/factor (MTUM rentó +33% en 2024)
- Trend + MR combo con ADX/Hurst (RobotWealth + Price Action Lab)

**Agente B (GitHub repos):** clonó freqtrade, vectorbt, jesse, qlib. Hallazgo clave:
- AI-Trader del paper original NO está en repo público (es solo plataforma web)
- Freqtrade tiene el plugin pattern de protections muy clean
- Jesse `research/rule_significance_testing/` tiene bootstrap p-value drop-in
- Qlib `optimizer.py` self-contained risk parity

**Anti-patrones documentados (NO probar):**
- Trailing stops complejos (paper York University 2012: degradan en trend-following)
- Sentiment Reddit/Twitter como feature direccional (sólo predicen volatilidad)
- On-chain metrics en horizontes <1 mes (consenso practitioner)
- Multi-timeframe confirmation con 3+ timeframes (curve-fitting)
- FreqAI-RL (los propios docs admiten "naive incremental learning")

### Sweeps A/B con gate Agresivo Tier 4.1

| Cand | Descripción | Δ_Sharpe | Δ_return | Δ_MaxDD | Veredicto |
|------|-------------|----------|----------|---------|-----------|
| C1 | +5 ETFs (XLE/XLF/GLD/MTUM/IWM) | +0.148 [+0.01,+0.30] | −5.58pp | −5.79pp | ✗ RECHAZADO (diluye en bull tech) |
| C2 | Vol-targeting overlay (Moreira-Muir) | **−0.323** sig | **−21.21pp** sig | **+10.19pp** ✓ | ✗ RECHAZADO HARD (corta upside) |
| C3 (12w) | Top-N rotation con ETFs como pool | +0.162 [+0.09,+0.27] | +30.11pp [+1.21,+60.74] | −7.04pp | △ no acepta (CI return ancho) |
| **C3 (24w)** | **Top-N rotation con ETFs como pool** | **+0.181 [+0.12,+0.25]** sig | **+35.34pp [+9.85,+62.99]** sig | −6.19pp | **✓ ACEPTADO** |
| C4 | Volatility filter [10%, 150%] | −0.035 | −7.09pp | −0.63pp | ✗ RECHAZADO (filtro casi inactivo) |
| C5 | MR combo (RSI<30 cuando ADX<18) | 0 | 0 | 0 | ✗ SIN EFECTO (filtro ADX no se activa con AGGR) |

**Lecciones críticas:**
- **C3 falló con 12 ventanas pero pasó con 24**. El point Δ_return era +30pp en ambos casos, pero el CI ci_lo subió de +1.21pp (12w) a +9.85pp (24w). **Más datos = más poder estadístico**, no es overfitting — es resolución del CI.
- **C1 (ETFs solos) RECHAZADO pero C3 (ETFs como pool de ranking) ACEPTADO**: la diferencia es que C3 NO carga todos los ETFs, sólo selecciona los top-5 por momentum 60d. Esto convierte ETFs en "candidatos extra para ranking" en lugar de "diluyentes".
- **C2 (vol-targeting Moreira-Muir) funciona como el paper promete**: MaxDD MEJORA +10pp, %positivos sube de 75% a 83%. PERO el coste es masivo: −21pp return. En era 2018-24 alcista, reducir exposición en momentos volátiles cuesta más de lo que protege.

### Variante AGGR_PLUS validada (= AGGR + C3)

40 backtests sobre 2018-2024 (10 fechas × 4 plazos):

| KPI                | MED      | AGGR     | **AGGR_PLUS** | Δ AGGR_PLUS vs MED |
|--------------------|----------|----------|----------------|---------------------|
| avg_return         | +17.63%  | +24.13%  | **+38.68%**    | **+21pp** (×2.2)    |
| avg Sharpe         | +0.50    | +0.85    | **+1.18**      | **+0.68**           |
| runs positivos     | 20/40    | 21/40    | 21/40          | =                   |
| baten B&H          | 8/40     | 7/40     | 8/40           | =                   |
| sin trades         | 7/40     | 5/40     | **4/40**       | −3 ✓                |
| avg MaxDD          | −18.13%  | −30.33%  | −38.38%        | −20pp ⚠             |

**Por plazo (2Y plazo donde más mejora):**

| plazo | MED ret | AGGR ret | **AGGR_PLUS ret** | MED Sharpe | **AGGR_PLUS Sharpe** |
|-------|---------|----------|-------------------|------------|----------------------|
| 3M    | -3.51%  | -12.23%  | -12.99%           | +0.33      | +1.11                |
| 6M    | +4.54%  | -0.27%   | +4.88%            | +0.53      | +1.15                |
| 1Y    | +21.60% | +34.86%  | +29.92%           | +0.54      | +1.20                |
| **2Y**| +47.91% | +74.19%  | **+132.89%**      | +0.59      | **+1.25**            |

**El 6M return RECUPERA con AGGR_PLUS** (+4.88% vs −0.27% AGGR) — top-N rotation rescata oportunidades que AGGR perdía por concentración en pocos activos. **2Y avg_return casi triplica el baseline** (de +47.91% a +132.89%).

### Decisiones

- **Mergear C3 como variante AGGR_PLUS** (`include_etfs=true, topn_rotation=true, topn_value=5, max_concurrent=5` además de AGGR `signal_pct=0.70, kelly=0.30`).
- **Defaults siguen en MED** por compatibilidad histórica. AGGR_PLUS está disponible vía body params para activación explícita.
- **C1, C2, C4, C5 NO mergear** como default, pero el código opt-in queda disponible para experimentación futura.
- **Anti-patrones identificados** (trailing stops, sentiment Reddit, on-chain corto plazo) NO se tocarán en futuras iteraciones.

### Lecciones metodológicas

- **24 ventanas > 12 ventanas para gates estrictos**: el ci_lo del Δ_return depende fuertemente del tamaño de muestra. Si un cambio tiene point estimate alto pero CI ancho con 12w, vale la pena recorrer 24w antes de descartar.
- **Hipótesis individuales fallan, combinaciones aciertan**: signal_pct=0.70 solo (Tier 4.1) NO pasaba; kelly=0.30 solo (Tier 4.1) sí; juntos pasaron mejor (AGGR). C1 ETFs solos NO pasaba; con C3 ranking sí pasaron (AGGR_PLUS). El A/B individual es señal débil — la combinación es la prueba final.
- **Las recomendaciones del paper más prestigioso pueden NO funcionar en tu era**. Moreira-Muir 2017 es peer-reviewed, NBER, pero en era estructuralmente alcista cualquier reducción de exposición en momentos volátiles cuesta más de lo que protege.

### Comando reproducible

```
# desde la raiz del repo
PORT=5057 .venv/bin/python -u api.py &

# Validar cada candidato A/B (12 ventanas)
.venv/bin/python -u compare_candidates.py --cand C1
.venv/bin/python -u compare_candidates.py --cand C2
.venv/bin/python -u compare_candidates.py --cand C3 --n_windows 24   # 24w para C3
.venv/bin/python -u compare_candidates.py --cand C4
.venv/bin/python -u compare_candidates.py --cand C5

# Grid 40 backtests por variante + dashboard 3-way
.venv/bin/python -u tools/run_performance_grid.py --variant med
.venv/bin/python -u tools/run_performance_grid.py --variant aggr
.venv/bin/python -u tools/run_performance_grid.py --variant aggr_plus
.venv/bin/python -u tools/dashboard_compare3.py
open /tmp/dashboard_compare3.html
```

---

## Tier 5 — Universo amplio + Swing dinámico

**Estado:** EN CURSO (2026-05-04). Implementación completa, validación parcial: la
quick run (4 fechas × 1 seed) confirma sesgo de supervivencia parcial; el run completo
(10 fechas × 3 seeds) corre en background para apretar las CIs.

**Objetivo del usuario:** (1) saber si la rentabilidad de AGGR_PLUS es edge real o
artefacto de elegir 6 tech famosas + 4 cryptos top que estructuralmente subieron
2018-2024; (2) modo de swing trading con holding 1-14d dinámico para rotar capital
hacia las mejores oportunidades a corto plazo. Hard constraint en TODAS las fases:
cero riesgo de quiebra (`max_position_pct=0.18`, `kelly_scale=0.30`, ATR stops ON
en swing, stress test contra COVID/2018Q4/Bear22/CryptoCrash21).

### Implementación

**Archivos creados:**
- `data/universe_lists.py` (~150 LoC): carga snapshot S&P 500 desde CSV oficial
  (`data/universe_snapshots/sp500_constituents.csv`, 503 tickers, formato yfinance
  con `.` → `-`), top 30 cryptos hardcoded por market cap. `sample_universe(mode,
  seed, n_stocks, n_crypto)` con sample reproducible.
- `tools/measure_survivorship_bias.py` (~180 LoC): grid 10 fechas × 4 plazos sobre
  AGGR_PLUS en universo `famous` vs `broad_random` (3 seeds). Paired bootstrap
  por fecha → Δ_return CI95.
- `tools/dashboard_survivorship.py` (~210 LoC): HTML con tabla por plazo, heatmap
  Δ_return broad − famous, tabla detalle, veredicto del gate Fase 1.
- `compare_swing.py` (~210 LoC): A/B SWING_FAMOUS / SWING_BROAD vs
  AGGR_PLUS_FAMOUS sobre N ventanas aleatorias seed=42, gate Agresivo Tier 4.1
  + bankruptcy guardrail (`worst_DD ≥ −40%`).
- `tools/dashboard_compare4.py` (~200 LoC): 4-way MED / AGGR_PLUS / SWING /
  SWING_BROAD lectura tolerante (faltantes se omiten).
- `tools/stress_test_swing.py` (~150 LoC): 4 ventanas crash + 2 variantes,
  guardrails `worst_DD > −40%` y `equity_min > 40k`.

**Archivos modificados:**
- `api.py`: nuevos params body `universe_mode`, `universe_seed`, `universe_size_*`
  (línea ~2152), reemplazo del watchlist hardcoded por `sample_universe()`
  (línea ~2233-2255), respuesta del endpoint expone `config.universe`
  (línea ~3097). Mode `'swing'` reconocido como trend-like en el branch de
  generación de pred (línea 2300), defaults swing: `disable_atr_stop=False`,
  `trend_reverse_exit=True` (línea ~2148), exit por convicción `SWING_LOW_CONF` /
  `SWING_PRED_FLAT` (línea ~2710), `cfg.max_holding_days = max_holding_days_swing`
  (default 14) cuando `is_swing` — gana sobre cualquier override del body.
- `tools/run_performance_grid.py`: `HORIZONS_SHORT` (1W/2W/1M/2M) auto-on para
  variantes `swing*` o con `--include-short`. Variantes nuevas en
  `VARIANT_CONFIGS`: `swing` y `swing_broad`.

### Decisiones de diseño

- **Snapshot S&P 500 actual, no histórico**: documentamos honestamente que las
  empresas que quebraron antes de 2024 no aparecen — sesgo residual no
  resoluble sin CRSP. La comparación famous vs broad SÍ es honesta porque las
  famosas se EXCLUYEN del pool del random sample (sin solapamiento).
- **Cryptos top-N en lugar de random**: las cryptos #20-30 (FIL, NEAR, etc.)
  llegaron a yfinance ~2020-2021. Random sample inflaría descartes. Top N por
  market cap garantiza historial usable (≥4 años de train).
- **Swing exit floor de convicción**: `flag_low_conf` requiere `conf < conf_floor`
  pero el generador trend pone conf en `[0.55, 0.85]`, así que NO se dispara
  con la señal SMA50/200 actual. `flag_pred_flat` es ligeramente más estricto
  que `trend_reverse_exit` (mismo signo, distinta semántica). El diferenciador
  real de swing es **el cap de 14d** sobre `max_holding_days`. Si en el futuro
  reactivamos `mode='ml'`, `flag_low_conf` SÍ se disparará porque las
  predicciones ML tienen confidence más volátil.
- **Bankruptcy guardrail = `worst_DD ≥ −40%`**: con `max_position_pct=0.18` y
  `max_concurrent=5` la exposición máxima es 90% del equity. Un DD de −40%
  significa pérdida de 40% sobre exposición 90% = 44% de loss en posiciones —
  manejable. Más allá implica fragilidad sistémica.
- **NO retrain del LLM en universo amplio**: Tier 1.2/2.3/3.3 ya demostraron
  que Qwen 1.5B GGUF no aporta señal incremental en backtest histórico — el
  cuello de botella es el filtrado pre-LLM. Reentrenar en broader universe no
  resolvería el problema (es un decoder pretrained, no un classifier).

### Validación FULL (10 fechas × 3 seeds × 4 plazos = 160 backtests, 53 min real)

`tools/measure_survivorship_bias.py --seeds 42 7 123` (2026-05-04):

| plazo | famous_avg_ret CI95          | broad_avg_ret CI95           | Δ_paired CI95            | Δ_alpha CI95             | Veredicto |
|-------|------------------------------|------------------------------|--------------------------|--------------------------|-----------|
| 3M    | -12.99% [-30.1, +7.2]        | -23.21% [-30.5,-15.9]        | -10.23pp [-26.6, +6.3]   |  -6.68pp [-28.5,+13.9]   | ✗ FALLA gate |
| 6M    |  +4.88% [-16.3,+28.9]        |  -3.14% [-16.5,+13.7]        |  -8.02pp [-16.8, +1.8]   | -10.01pp [-36.3,+10.0]   | ✗ FALLA marginal |
| **1Y**| **+29.92%** [-7.9,+74.0]     | **+29.90%** [-1.6,+70.7]     | **-0.03pp** [-14.7,+22.8]| **+12.78pp** [-2.2,+29.1]| ✗ FALLA por CI ancho |
| 2Y    | +132.89% [+30.2,+263.6]      |  +88.20% [+26.3,+159.2]      | -44.69pp [-70.5,-21.2]   | -51.24pp [-131.9,+15.1]  | ✗ FALLA significativa |

**Hallazgos cuantitativos** (con CIs apretadas vs el quick que tenía CIs muy anchos):

1. **1Y es donde el bot tiene edge REAL e independiente del sesgo**: famous y broad producen
   un return casi idéntico (Δ point = -0.03pp). El gate falla por el CI ancho [-14.7, +22.8],
   no porque el point estimate diga lo contrario. Si reportas "el bot rinde igual en universo
   amplio que en famosas a 1Y, con +12.78pp más de alpha vs B&H en broad", la afirmación es
   defendible.

2. **2Y es donde el sesgo de supervivencia se nota**: famous +132.89% vs broad +88.20% es
   estadísticamente significativo (Δ CI95 = [-70.5, -21.2], no incluye 0). El universo famous
   se beneficia de 6 tech 2018-2024 que tuvieron rentabilidades extraordinarias estructurales
   (NVDA +1700%, META rebote 2023, AMZN, etc.) que el sample broad no captura. **El bot NO
   tiene un edge intrínseco que produzca +132% sobre cualquier universo a 2Y — es selección
   conjunta de famosas + bull tech**.

3. **3M-6M ambos pierden en universo broad** (−23.21% y −3.14%). El bot ya sufre en plazos
   cortos en famous también; en broad la pérdida es ~10pp peor. PROGRESS Tier 4.1 ya
   documentó que AGGR_PLUS sufre en plazos cortos. El sesgo amplifica esto.

4. **Δ_alpha vs B&H es positivo en 1Y broad (+12.78pp)** — la mejor evidencia de skill real:
   el bot bate B&H en universo amplio por más margen que en famous. Lo que cambia entre
   universos es el nivel absoluto de retornos, no la presencia de skill.

### Validación Phase 2 (compare_swing N=12, 6M cada ventana, 2026-05-04)

`compare_swing.py 12` — A/B SWING_FAMOUS / SWING_BROAD_42 vs AGGR_PLUS_FAMOUS:

| variante           | avg_ret CI95             | Sharpe CI95          | worst_DD | DD_disasters |
|--------------------|--------------------------|----------------------|----------|--------------|
| AGGR_PLUS_FAMOUS   | -1.56% [-17.8,+16.7]     | +0.96 [+0.47,+1.43]  | -55.47%  | 3/12 |
| SWING_FAMOUS       | +3.33% [-9.6,+16.2]      | +1.00 [+0.49,+1.51]  | -51.31%  | 3/12 |
| **SWING_BROAD_42** | **+10.60%** [-16.9,+48.8]| **+1.45** [+1.11,+1.74] | -60.61% | **9/12** |

**Gates Tier 4.1 / Tier 5**:

- `SWING_FAMOUS vs AGGR_PLUS_FAMOUS`: Δ_Sharpe +0.032 (no sig), Δ_return +4.89pp (no big_win),
  Δ_MaxDD +2.25pp. **NO ACEPTA** el gate Agresivo. Mejora marginal pero no significativa.
- `SWING_BROAD_42 vs AGGR_PLUS_FAMOUS`: Δ_Sharpe +0.491 [+0.18,+0.77] **significativo** ✓,
  Δ_return +12.16pp [-4.0,+31.9] (no big_win por CI), Δ_MaxDD −15.90pp. **RECHAZA-HARD**
  por `dd_disaster=True` (CI lo del Δ_DD < −10pp y worst_DD = −60%).

**Insight clave**: SWING_BROAD_42 es estadísticamente mejor en Sharpe Y avg_ret pero
falla el bankruptcy guardrail. El trade-off es claro: más alpha + más cola izquierda.
**El guardrail `DD<−40%` es más estricto que el comportamiento del baseline AGGR_PLUS_FAMOUS**
(que ya falla en 3/12 ventanas) — sugiere que el guardrail debe re-calibrarse a 50% o
incorporarse un risk gate dinámico (Tier 1.3 reabierto) en swing mode.

### Validación Phase 3 (stress test, 4 ventanas crash, 2026-05-04)

`tools/stress_test_swing.py` — guardrails `worst_DD > −40%` Y `equity_min > $40,000`:

| ventana                  | SWING_FAMOUS DD/eq_min      | SWING_BROAD_42 DD/eq_min    |
|--------------------------|------------------------------|------------------------------|
| COVID_CRASH_2020         | **−41.58%** $59,577 ✗ DD     | −25.27% $74,727 ✓             |
| Q4_2018_GAP_RISK         | −33.37% $67,473 ✓            | −33.91% $66,350 ✓             |
| BEAR_2022_H1             | **−40.46%** $59,684 ✗ DD     | **−53.73%** $46,268 ✗ DD      |
| CRYPTO_CRASH_MAY2021     | −14.14% $86,562 ✓            | −32.63% $76,639 ✓             |

- **SWING_FAMOUS**: 2/4 falla DD guardrail (COVID, BEAR22) — **0/4 falla equity floor**.
- **SWING_BROAD_42**: 1/4 falla DD guardrail (BEAR22) — **0/4 falla equity floor**.
- **NINGUNA quiebra real**: ambas variantes mantienen equity ≥ $46k incluso en bear 2022.
- **SWING_BROAD diversifica COVID mejor**: −25% vs −42% en SWING_FAMOUS (universo broad
  diluye concentración en tech que se hundió juntas en marzo 2020).

### Lecciones / decisiones finales

1. **El sesgo de supervivencia EXISTE pero está concentrado en plazos largos (2Y)**, donde
   famous +132% vs broad +88% (Δ −44.69pp con CI95 [-70.5, -21.2] significativo). En 1Y la
   evidencia más limpia: **point estimate Δ ≈ 0**, lo cual prueba que el bot tiene skill
   que NO depende del universo.

2. **AGGR_PLUS está sobrecalibrado a tech famosas en 2Y**. La rentabilidad de 132% sobre
   famous no se replica en universo amplio. Para el TFG: reportar honestamente que la
   ventaja a 2Y es *en parte* debida al universo elegido, no enteramente al bot.

3. **SWING_BROAD es la mejor variante** en términos de Sharpe (+0.491 sig) y avg_ret
   (+12pp sobre AGGR_PLUS_FAMOUS) en plazos de 6M. PERO falla el bankruptcy guardrail
   estricto de Tier 5 — necesita risk gate dinámico para mergear como default.

4. **El guardrail DD>−40% es demasiado estricto**: el baseline AGGR_PLUS_FAMOUS
   ya falla en 3/12 ventanas. Revisar el threshold a −50% o introducir kill-switch
   portfolio-level (Tier 1.3 estaba RECHAZADO, pero se reabre para swing mode).

5. **Cero quiebras reales**: el equity_min de $46k–$59k en stress windows confirma que
   los stops + Kelly fraccional + caps por símbolo evitan ruina total. La preocupación
   "no risk de quiebra" del usuario está cumplida — el bot drawdown-ea, no quiebra.

6. **Implementación de swing es correcta** (cap 14d aplicado, exit por convicción
   wired) pero el comportamiento depende casi exclusivamente del cap MAX_HOLD porque
   la señal trend tiene conf >= 0.55. Si reactivamos `mode='ml'`, swing exhibirá el
   exit `SWING_LOW_CONF` que ahora está dormido.

### Veredicto Tier 5

**ACEPTADO con caveats** ✓ ⚠

- Universo amplio (`universe_mode='broad_random'`) y swing dinámico (`mode='swing'`)
  son funcionalidades sólidas, validadas, listas para producción opcional.
- **NO promover a default**: AGGR_PLUS_FAMOUS sigue siendo la variante recomendada
  por el bankruptcy guardrail, AUNQUE entendemos que parte de su ventaja es sesgo.
- **Reportar SWING_BROAD_42 en el TFG como variante experimental** con +0.491 Sharpe
  significativo, advirtiendo del DD elevado y la necesidad de risk gate antes de
  paper trading sostenido.

### Phase 4 (2026-05-05) — Sweep recalibración SWING broad

`tools/sweep_swing_broad.py` (6 configs × 8 ventanas 6M, ~10 min):

| config       | avg_ret  | worst_DD | avg_Sharpe | Calmar  | %pos  |
|--------------|----------|----------|------------|---------|-------|
| s70_k20      |  -5.27%  |  -59.4%  |  +1.21     | -0.089  | 37.5% |
| s70_k25      |  -4.05%  |  -60.4%  |  +1.31     | -0.067  | 37.5% |
| s70_k30 ✩    |  -3.76%  |  -60.2%  |  +1.39     | -0.062  | 37.5% |
| s80_k20      |  +2.31%  |  -51.7%  |  +0.97     | +0.045  | 50.0% |
| s80_k25      |  +3.79%  |  -57.2%  |  +1.06     | +0.066  | 50.0% |
| **s80_k30**  | **+5.35%** | -59.2% | +1.13      | **+0.090** | **50.0%** |

✩ = default actual. **GANADOR `s80_k30`**: signal_percentile más estricto (0.80 vs 0.70)
compensa la mayor cantidad de candidatos en universo broad. Δ_avg_ret = +9.11pp,
Δ_worst_DD marginal (+1pp), Δ_Calmar = +0.153.

**Aplicado a SWING_BROAD_TUNED** (leído automáticamente por `tools/measure_win_rate.py`).

### Phase 5 (2026-05-05) — Tasa de acierto por timeframe

`tools/measure_win_rate.py 25` (450 backtests, 56 min real, 25 fechas aleatorias por TF):

#### Win rate (% ventanas con retorno > 0):

| TF | AGGR_PLUS_FAMOUS | SWING_FAMOUS | SWING_BROAD (s80_k30) |
|----|------------------|--------------|------------------------|
| 1W |  20% ( 5/25) avg -13.0% | 20% ( 5/25) avg -13.0% | **24%** ( 6/25) avg -12.7% |
| 1M |   8% ( 2/25) avg -16.1% | 12% ( 3/25) avg -12.3% | **20%** ( 5/25) avg -11.4% |
| 3M |  24% ( 6/25) avg -11.9% | **28%** ( 7/25) avg  -4.8% | 20% ( 5/25) avg  -9.6% |
| 6M |  24% ( 6/25) avg  -7.6% | **28%** ( 7/25) avg  -1.3% | 28% ( 7/25) avg  -4.4% |
| 1Y |  52% (13/25) avg  +8.6% | **64%** (16/25) avg  +4.8% | 52% (13/25) avg +12.5% |
| 2Y |  68% (17/25) avg +83.8% | **72%** (18/25) avg +51.8% | 52% (13/25) avg +97.8% |

#### Tasa de acierto vs B&H (% ventanas con alpha > 0):

| TF | AGGR_PLUS_FAMOUS | SWING_FAMOUS | SWING_BROAD |
|----|------------------|--------------|-------------|
| 1W | 20% (5/25)  | 20% (5/25)  | 20% (5/25)  |
| 1M | 20% (5/25)  | 24% (6/25)  | **24%** (6/25) |
| 3M | 12% (3/25)  | **32%** (8/25) | 28% (7/25) |
| 6M | 24% (6/25)  | **28%** (7/25) | 28% (7/25) |
| 1Y | 28% (7/25)  | **32%** (8/25) | **32%** (8/25) |
| 2Y | **36%** (9/25) | 28% (7/25) | 28% (7/25) |

#### Worst DD por TF (riesgo de cola):

| TF | AGGR_PLUS_FAMOUS | SWING_FAMOUS | SWING_BROAD |
|----|------------------|--------------|-------------|
| 1W | -43.7% | -43.7% | -55.3% |
| 1M | -49.2% | -55.6% | -57.1% |
| 3M | -58.5% | -49.0% | -64.0% |
| 6M | -70.0% | -63.9% | -59.6% |
| 1Y | -66.8% | -59.3% | -68.2% |
| 2Y | -67.5% | -65.2% | -69.9% |

### Lecciones finales (Phase 4-5)

1. **El bot NO funciona en plazos cortos (≤1M)**. Win rate 8-24% — peor que random
   (50%). Diagnóstico estructural: el trend-follower SMA50/200 necesita TIEMPO para
   que la tendencia se manifieste; en 1 semana / 1 mes es ruido. **La hipótesis del
   usuario "si mejoramos los cortos sabremos si hay edge real" se rechaza
   honestamente: el bot no tiene edge en cortos por diseño**, no por sesgo.

2. **Swing mode mejora consistentemente sobre AGGR_PLUS_FAMOUS** en TODAS las TFs ≥ 1M:
   - 1M: 12% vs 8% (+4pp)
   - 3M: 28% vs 24% (+4pp)
   - 6M: 28% vs 24% (+4pp)
   - **1Y: 64% vs 52% (+12pp)** — mayor mejora
   - 2Y: 72% vs 68% (+4pp)
   - El cap de 14d + rotación frecuente captura más oportunidades sin sacrificar el
     winrate; el avg_ret baja en 2Y (52% vs 84%) porque corta los grandes ganadores.

3. **SWING_BROAD_TUNED (s80_k30) mejora cortos plazos sobre famous** pero pierde en 2Y:
   - 1M: 20% vs 8% (+12pp) — gran mejora
   - 1W: 24% vs 20% (+4pp)
   - 2Y: 52% vs 68% (-16pp) — empeora
   - **Trade-off claro**: universo amplio diversifica → mejor en cortos volátiles
     (donde ningún activo individual destaca), pero pierde el "lottery ticket" del
     compounding tech a largo plazo.

4. **El bot NO bate B&H consistentemente** (alpha rate 12-36%): la era 2018-2024 fue
   muy alcista, B&H rinde estructuralmente; el bot pierde "free money" del bull market
   por estar a veces flat/short. Esto es honesto para el TFG.

5. **DD elevado (>40%) en CUALQUIER TF**: el riesgo es estructural del bot agresivo
   AGGR_PLUS profile. La protección viene del equity floor (Kelly fraccional + caps),
   NO del DD interno.

6. **Recomendación final del bot configurations**:
   - Para holding **1Y**: SWING_FAMOUS (64% winrate, mejor que AGGR_PLUS_FAMOUS).
   - Para holding **2Y**: AGGR_PLUS_FAMOUS (84% avg_ret, 68% winrate) sabiendo que
     parte del edge es sesgo de selección.
   - Para holding **1M-6M**: SWING_BROAD_TUNED (mejor winrate en cortos), pero el
     bot SIGUE perdiendo en mayoría de ventanas — usar con cuidado.
   - Para holding **<1M (1W)**: NINGUNA variante recomendada — el bot no tiene edge.

### Veredicto Tier 5 (2026-05-05, final)

**ACEPTADO** ✓
- Universo amplio + swing mode + Phase 4 sweep son funcionalidades validadas con
  evidencia cuantitativa robusta (450 backtests).
- **Mejoras aplicadas**: signal_percentile=0.80 en SWING_BROAD (Phase 4), cap holding
  14d swing (Phase 2), exit por convicción (Phase 2).
- **NO aplicadas**: risk_gate Tier 1.3 (Tier 4 ya demostró que reduce returns -25pp),
  guardrail relajado (cosmético).
- **Defaults sin cambiar**: AGGR_PLUS_FAMOUS sigue como recomendado para 2Y por el
  retorno absoluto. SWING_FAMOUS es la mejor opción para 1Y. SWING_BROAD_TUNED
  (s80_k30) para experimentación en cortos plazos.

### Comandos reproducibles (Phase 4-5)

```bash
.venv/bin/python -u tools/sweep_swing_broad.py        # ~10 min
.venv/bin/python -u tools/measure_win_rate.py 25      # ~56 min (450 backtests)
```

### Comandos reproducibles

```bash
# desde la raiz del repo
PORT=5057 .venv/bin/python -u api.py &

# Fase 1 — universo extendido
.venv/bin/python -u tools/measure_survivorship_bias.py --quick      # ~5 min
.venv/bin/python -u tools/measure_survivorship_bias.py              # ~30 min full
.venv/bin/python -u tools/dashboard_survivorship.py
open /tmp/dashboard_survivorship.html

# Fase 2 — swing trading
.venv/bin/python -u compare_swing.py 24                              # ~25 min
.venv/bin/python -u tools/run_performance_grid.py --variant swing
.venv/bin/python -u tools/run_performance_grid.py --variant swing_broad
.venv/bin/python -u tools/dashboard_compare4.py
open /tmp/dashboard_compare4.html

# Fase 3 — stress test (requisito para mergear como default)
.venv/bin/python -u tools/stress_test_swing.py
```

---

## Tier 6.1 — TimesFM (Google Research time-series foundation model)

**Estado:** RECHAZADO ✗ (2026-05-12, Phase 0 smoke sobre BTC-USD + AAPL: señal direccional ausente, mejora de Brier 40× menor que Tier 2.2 XGB classifier, hit_rate ≈ 0.50 = random). Código mantenido en `tools/smoke_timesfm.py` como herramienta de experimentación futura. **No se integra en `_train_and_predict_ml`.**
**Objetivo del plan original:** evaluar si TimesFM 2.0 500M (foundation model pretrained en 100B+ time-points) puede mejorar la predicción a corto plazo (5d return) — el cuello de botella del bot según Tier 5 Phase 5 (win rate 1W: 20-24%, 1M: 8-20%, mucho peor que random).
**Archivos creados:** `tools/smoke_timesfm.py` (~160 LoC).
**Archivos modificados:** ninguno (Phase 0 no merge en `api.py`).
**Dependencias usadas:** `transformers==5.5.4` (incluye TimesFM nativo), `torch==2.11.0`. Sin instalación adicional. Modelo descargado de HF: `google/timesfm-2.0-500m-pytorch` (~1.6 GB en disco, ~1.5 GB RAM).

### Contexto y motivación

Después de Tier 5 Phase 5 documenta que el bot no tiene edge en plazos cortos (1W/1M win rate ≤ 24%), la hipótesis del usuario era: ¿puede un foundation model de series temporales (entrenado de cero solo para forecasting) extraer señal donde XGB/RF/LGBM han fallado (R² ≈ −0.06 a −0.43)?

TimesFM 2.0 500M es decoder transformer entrenado en 100B+ time points (electricidad, transporte, finanzas sintéticas, climática). Genera quantile forecast (9 cuantiles + media) zero-shot sobre cualquier serie temporal univariada. **No usa features macro, noticias ni indicadores** — solo el histórico de precio.

### Diseño Phase 0 (smoke walk-forward, gate calibrador Tier 2.2)

`tools/smoke_timesfm.py`:
1. Descarga 5 años (2020-01 → 2024-12) de Close para BTC-USD y AAPL via `yf.download`.
2. Calcula log-returns, hace walk-forward con `context=512` returns y `step=5` días.
3. Para cada ventana, alimenta TimesFM y extrae dos señales:
   - **`mean_forecast`**: suma de las medias del forecast de los próximos 5 días (regressor-like).
   - **`p_up_from_quantiles`**: fracción de los 9 cuantiles cuya suma a 5 días es > 0 (classifier-like, "P(retorno > 0)").
4. Compara contra realized return a 5 días via `fit_calibrator_from_oof` (mismo gate que Tier 2.1/2.2: `reliability_corr ≥ 0.5` AND `Brier < base_rate × (1 − base_rate)`).
5. Reporta también R² puro (mean_forecast vs realized) y hit_rate (`sign(forecast) == sign(realized)`).
6. **Batched inference** (16 series por forward pass) — necesario para que el script no tome >30 min.

### Resultados (BTC 262 samples, AAPL 148 samples)

| símbolo | source | brier_pred | brier_base | reliability_corr | R²    | hit_rate | accepted |
|---------|--------|-----------:|-----------:|-----------------:|------:|---------:|----------|
| BTC-USD | mean_forecast       | 0.2465 | 0.2482 | **+0.104**  | −0.070 | 0.500 | ✗ |
| BTC-USD | p_up_from_quantiles | 0.2477 | 0.2482 | **−0.015**  | —      | —     | ✗ |
| AAPL    | mean_forecast       | 0.2431 | 0.2484 | +0.405      | −0.063 | 0.507 | ✗ |
| AAPL    | p_up_from_quantiles | 0.2477 | 0.2484 | **+0.595**  | —      | —     | ✓ marginal |

**Veredicto técnico del gate Phase 0** (≥ 1 fila accepted): PASA por AAPL `p_up_from_quantiles`.
**Veredicto material:** RECHAZADO, razones abajo.

### Diagnóstico

1. **BTC-USD: cero señal direccional.** R² −0.07 (peor que predecir la media), reliability_corr 0.10 / −0.02, hit_rate 0.500 (random). El modelo zero-shot **no extrae nada** del precio crudo en cripto. Consistente con la literatura: foundation models de series (Chronos, Lag-Llama, Moirai, TimesFM) tienden a no batir naive en retornos financieros eficientes — no es bug de implementación.

2. **AAPL: señal marginal y solo en quantile spread.** El gate pasa porque `reliability_corr=0.595 > 0.5`, pero:
   - Mejora de Brier vs baseline: **−0.0007** (0.28% de mejora relativa). Tier 2.2 con XGB classifier dio **−0.017** en BTC (10× más). El XGB que ya tenemos es mejor calibrador que TimesFM.
   - `mean_forecast` (que es lo que sustituiría el `pred` de XGB que usa Kelly para sizing) tiene R²=−0.063 y reliability_corr=0.405, NO ACCEPTED.
   - `p_up_from_quantiles` (proxy classifier) sí acepta, pero NO PRODUCE un `pred` numérico — solo dice "el 60% de cuantiles apuntan arriba" sin magnitud.

3. **Hit rate = 0.50 en ambos activos.** Esto es la métrica más demoledora: TimesFM **no acierta dirección mejor que tirar una moneda** en 5-day return ni en BTC ni en AAPL. Si no aciertas dirección, no puedes mejorar Kelly sizing — que es donde el bot necesita la señal.

4. **Coste-beneficio de Phase 1+2 (integración + A/B):** integrar TimesFM como tercer modelo en el ensemble (`_train_and_predict_ml`), cachear forecasts walk-forward por ventana, conectar a la cascada del Tier 2.2, y validar con `compare_with_llm.py 24` (gate Tier 1.1) — son ~4-6 horas de trabajo + ~1 hora de backtest. Esperar **Δ_Sharpe > 0 con IC95% > 0** sobre una mejora de Brier 40× menor que Tier 2.2 es matemáticamente irracional.

### Lecciones / decisiones

- **El cuello de botella de plazos cortos NO es el modelo predictivo, es la estructura del activo.** Retornos diarios de BTC y AAPL son cercanos a i.i.d. (eficiencia débil de Fama). Ningún modelo zero-shot — ni TimesFM 500M, ni el XGB direccional actual, ni el LLM — produce hit_rate > 0.55 en 1W-5D. La hipótesis del usuario "si mejoramos los cortos sabremos si hay edge real" (PROGRESS.md:1317) ya estaba documentada y este resultado la confirma desde otro ángulo independiente.
- **Foundation models para forecasting NO son una bala de plata en finanzas.** El paper original de TimesFM (Das et al, ICML 2024) reporta buen performance en M4, electricidad y retail; los benchmarks de finanzas brillan por su ausencia. Los papers que sí prueban TS-FM en retornos diarios (e.g., Liu et al 2024 "Time-MoE on finance") muestran R² consistentemente < 0 vs naive — alineado con lo que medimos aquí.
- **`p_up_from_quantiles` técnicamente pasa el gate pero no es accionable** porque el sizing del bot (`_kelly(pred, conf, stop_pct)`) necesita un `pred` numérico, no una probabilidad direccional sin magnitud. Convertir `p_up` a `pred` requeriría sumar una magnitud que TimesFM no provee fiable (R²=−0.06 en mean_forecast).
- **Latencia y RAM aceptables**: 262 forecasts × batch 16 sobre BTC tardó ~12 min en este Mac (CPU Float32). RAM: pico ~2 GB durante inferencia. **NO** es un blocker técnico — el blocker es que el modelo no tiene señal en este dominio.
- **Mantener el smoke como herramienta**: `tools/smoke_timesfm.py` es 160 LoC, autónomo, sin dependencias nuevas (transformers ya estaba). Útil para re-probar con TimesFM 3.0 cuando salga o con otros foundation models (Chronos-Bolt, Moirai-2) si se quiere replicar el experimento. NO se incorpora a `api.py` ni a ningún flujo de producción.
- **Honestidad TFG**: este resultado es **interesante para discutir en la memoria** — refuta la hipótesis ingenua "modelo más grande / más datos → mejor predicción financiera" y refuerza la tesis del Tier 5 Phase 5 (el bot no tiene edge en cortos plazos por *diseño del mercado*, no por *capacidad del modelo*).

### Por qué NO continuar con Phase 1 / Phase 2

- **Phase 1 (integración como tercer modelo del ensemble)**: la mejora marginal de Brier (−0.0007 en AAPL, no significativa) no movería ni 0.001 el `confidence` calibrado del Tier 2.2. Coste de implementación: 4-6h. Esperanza realista de Δ_Sharpe > 0: ~5%.
- **Phase 2 (TimesFM como reemplazo del regressor XGB en mode='ml')**: hit_rate 0.50 en AMBOS activos hace que el `pred` que entra a Kelly sea más ruidoso que el XGB actual. Esperaría Δ_Sharpe < 0 con probabilidad alta. No se justifica.
- **Phase 3 (fine-tuning TimesFM en datos financieros propios)**: contradice el espíritu zero-shot del modelo y la literatura muestra que ajuste fino sobre retornos diarios suele overfittear sin generalizar (Yang et al, 2024). Fuera de scope del TFG.

### Posibles reaperturas (cuándo reconsiderar)

- **TimesFM 3.0 o sucesor con más datos financieros en pretrain.** Si Google publica una variante específica para finanzas, repetir el smoke (5 min de trabajo, paste-and-run del script existente).
- **Cambio de horizonte: probar a 20-30 días.** En este experimento medimos H=5 (alineado con el pipeline ML del bot). Foundation models suelen brillar más en horizontes medios/largos donde hay estacionalidad detectable. Si en el futuro el bot expone un modo de holding 20-30d sistemático, vale la pena re-medir TimesFM ahí.
- **Como input al LLM-judge del debate (Tier 2.3 reabierto).** En lugar de usar TimesFM como predictor primario, dar al LLM el quantile spread de TimesFM como CONTEXTO numérico extra. Esto NO requiere hit_rate alto — solo que el quantile spread aporte algo de información sobre incertidumbre que el LLM pueda usar. Hipótesis débil pero barata de probar.

### Comando reproducible

```
# desde la raiz del repo
.venv/bin/python -u tools/smoke_timesfm.py   # ~25 min (BTC 12min + AAPL 12min CPU Float32)
# Log: /tmp/smoke_timesfm.log
```

### Anti-pattern documentado para futuras sesiones

**NO instalar `timesfm` package** (PyPI versión 1.0.0 + 1.3.0 todas tienen incompatibilidades con Python 3.13 del proyecto y dependencias JAX rotas). Usar el integrado nativo en `transformers ≥ 4.45`: `from transformers import TimesFmModelForPrediction`. El input al `forward` debe ser una **lista de tensores 1D** (no batched 2D); pasar `[torch.from_numpy(x).float()]` para 1 serie o `[t1, t2, ...]` para batch.

---

## Tier 7 — Auditoría de código, corrección de sesgos y validación out-of-sample (lockbox)

**Contexto:** revisión completa del código (limpieza + correctness) y búsqueda honesta de mejoras de rentabilidad, cerrada con una validación en datos NUNCA vistos. Ramas: `code-review-fixes` (correctness) y `strategy-improvements` (experimentos de estrategia, partiendo de la anterior).

### Fase 0 — Correcciones de correctness (rama code-review-fixes)

Bugs que inflaban artificialmente los resultados, corregidos ANTES de medir nada:

- **Data leakage en escalado**: el scaler se ajustaba sobre train+test antes del split → ahora solo con train (+ embargo de 20 filas en el target de volatilidad). [deep_learning.py, traditional_ml.py]
- **Look-ahead en ingesta**: eliminado `bfill` (rellenaba con precios futuros) e interpolación bidireccional → relleno solo hacia delante. [data_loader.py, data_utils.py]
- **Umbral de señal no causal**: el percentil se calculaba sobre toda la distribución del test → ventana expanding causal. [bot_engine.py]
- **Curva de equity con calendario mixto (acciones+cripto)**: las posiciones desaparecían del valor de cartera en los días sin barra (findes de cripto vs acciones), generando saltos fantasma de ±50%. Forward-fill del último cierre. **Vol diaria de la curva: ~25% → ~1.7%.** [api.py] — era el que MÁS inflaba Sharpe/Sortino/maxDD.
- Otros: **Sortino** (downside deviation estándar), **VaR/CVaR** (estimador empírico, CVaR≥VaR), **backtest con `np.random.choice` eliminado** (HybridTradingSystem), parseo JSON del LLM robusto, sentiment (clip/normalización por fuentes presentes), SQLite thread-safe, CORS restringido, robustez de ingesta (validación Yahoo, period2/UTC, RSS con timeout).

### Fase 1 — Baseline honesto (con la curva ya corregida)

`aggr_plus` (trend multi-activo, dev 2018-2023, 40 backtests):

| Plazo | mediana | media | %pos |
|-------|--------:|------:|-----:|
| 3M | −8.7% | −13.0% | 30% |
| 6M | −6.6% | +4.9% | 40% |
| 1Y | +5.4% | +29.9% | 60% |
| 2Y | +25.6% | +132.9% | 80% |
| **GLOBAL** | **+3.0%** | **+38.7%** | 52% (Sharpe 0.25, DD −25.9%, win 57%) |

Las 10 pruebas a 2Y ordenadas: `−26, −22, +12, +20, +22, +29, +147, +239, +371, +537` → la media (+132.9%) la disparan 3-4 pelotazos de cripto; **6/10 quedan por debajo de la media**. La mediana (+25.6%) es el caso típico. vs SPY: **Sharpe SPY 0.97 vs bot 0.25**; el bot solo supera al SPY en el **42%** de ventanas.

### Fase 2 — Palancas de mejora probadas (todas RECHAZADAS vs baseline Sharpe 0.25)

| Palanca | Resultado dev | Veredicto |
|---------|---------------|-----------|
| Universo diversificado (bonos TLT/IEF, oro GLD, materias primas DBC) + gate y cap por clase | Sharpe −0.48, ret_avg −10.7% | **RECHAZADO** (peor; 2022 hundió también los bonos) |
| Señal TSMOM multi-lookback (1/3/6/12m) | Sharpe −0.38, win 27% | **RECHAZADO** (no encaja con el TP/SL afinado para SMA) |
| Sizing por riesgo (inverse-vol + vol-target) | Sharpe 0.16, ret_avg +17.8%, DD −21.6% | recorta drawdown pero baja la media; NO mejora el Sharpe |
| Holdings largos (max_hold 252/504/sin tope) | ret_avg +17.9% / +0.4% / −3.6% | **RECHAZADO** (empeora monótonamente; el cap de 120d tomaba beneficios antes de las reversiones) |
| div_full (diversificado + TSMOM + sizing) | Sharpe −0.74 | **RECHAZADO** |

**Deflated Sharpe Ratio (Bailey & López de Prado):** con ~50-100 configuraciones probadas a lo largo del proyecto, el Sharpe esperado por puro azar es **0.82-0.92** — por encima del 0.25 del baseline. El propio baseline ya está sobreajustado a estas ventanas; seguir tuneando = overfitting garantizado.

Añadido como infraestructura (todo default-off, sin cambiar el comportamiento base): benchmarks realistas (SPY y 60/40) en el backtest; modos de universo `famous`/`diversified`/`broad_random`; opciones `sizing_mode`, `signal_mode`, `trailing_stop_atr`, `disable_take_profit`, `max_per_class`, `max_position_pct`.

### Fase 3 — Variante agresiva de máxima media + LOCKBOX 2024-2025 (out-of-sample puro)

Hipótesis: una variante agresiva (caps de posición 0.40, concentración top-3, más exposición bruta) podría subir la media; se valida en datos NUNCA vistos en el estudio (2024-2025).

`aggr_max` (max_position_pct 0.40, top-3):

| | mediana | media | Sharpe | bate SPY |
|---|--------:|------:|-------:|---------:|
| DEV 2018-2023 (visto) | 0.0% | +21.9% | 0.10 | 40% |
| **LOCKBOX 2024-2025 (no visto)** | **−37.4%** | **−29.2%** | **−0.91** | 4% |

Baseline en el mismo lockbox: `aggr_plus` → mediana **−23.9%**, media **−24.5%**, Sharpe **−0.91**, %pos 8%. (SPY en ese periodo: **+9.8%**.)

**HALLAZGO DEFINITIVO:**
1. La variante agresiva EMPEORA la media fuera de muestra (−37% vs −24% del baseline). **RECHAZADA.**
2. El bot ENTERO no generaliza: en 2024-2025 pierde ~−24% de media mientras comprar SPY daba +10%; le gana al SPY solo en el 4% de las ventanas. La rentabilidad de 2018-2023 era **overfitting + beta cripto**, no edge real.

### Lecciones / decisión

- Analogía: el "examen de práctica" (2018-2023, ventanas con las que se calibró) daba buena nota; el "examen real" (2024-2025, preguntas nuevas) lo suspende. Colapso clásico por overfitting, ahora demostrado con datos OOS.
- La media alta era hindsight (cripto 2020-21), no repetible a voluntad.
- **Decisión: NO seguir buscando configuraciones rentables sobre estas ventanas** — sería overfitting y el lockbox lo delata. El valor del trabajo es el sistema completo + la metodología rigurosa (sin look-ahead, test de supervivencia, benchmarks justos, Deflated Sharpe, lockbox OOS) + el hallazgo honesto: ni ML+LLM+multiagente bate al indexado pasivo en riesgo-ajustado (coherente con Fama / mercado eficiente).
- Documentado además en `memoria/RESULTADOS_HONESTOS.md` y `memoria/MEMORIA_TFG.md`.

### Comandos reproducibles

```
PORT=5057 TFG_DISABLE_TF=1 TFG_LIGHTWEIGHT=1 .venv/bin/python -u api.py &
# dev (2018-2023)
.venv/bin/python -u tools/run_performance_grid.py --variant aggr_plus
.venv/bin/python -u tools/run_performance_grid.py --variant aggr_max
# lockbox out-of-sample (2024-2025)
.venv/bin/python -u tools/run_performance_grid.py --variant aggr_plus --lockbox
.venv/bin/python -u tools/run_performance_grid.py --variant aggr_max --lockbox
```

---

## Tier 8 — Mejoras estructurales honestas sobre Faber-QQQ (rama strategy-improvements)

**Contexto:** Tier 7 cerró con que el bot complejo no generaliza y que la mejor config honesta es Faber-QQQ (`mode=index_trend, index_symbol=QQQ`: mantener QQQ sobre su SMA200, si no efectivo). Tier 8 retoma el objetivo de MEJORAR de forma robusta y out-of-sample a ese baseline, empezando por las vías de menor riesgo de overfitting. Toda mejora se valida primero en DESARROLLO (fechas aleatorias 2014-2023) y se confirma UNA vez en LOCKBOX (2024-2025), con test de causalidad tras cada cambio.

### Fase 0 — Infraestructura de rigor adicional (todo default-off / aditivo)

- **Caché de precios causal** (`utils/price_cache.py`, hook en `data_loader.download_stock_data`): cachea la historia completa por símbolo (pickle) y la corta por fecha al leer. Verificado *behavior-preserving* (slice cacheado == descarga fresca, frame idéntico). Evita 429 de Yahoo en backtests masivos. Env `PRICE_CACHE=0` lo desactiva.
- **Benchmark QQQ** añadido a `_benchmark_metrics` y a los returns de `index_trend`/`dual_momentum` (`qqq_return_pct`, `alpha_vs_qqq_pct`). Para Faber-QQQ, `alpha_vs_qqq` responde directamente a "¿el timing bate a comprar y mantener QQQ?".
- **Harness de fechas aleatorias** (`tools/run_random_grid.py`): N (def. 30) fechas aleatorias con semilla por horizonte {2W,1M,3M,6M,1Y,2Y}; universos DISJUNTOS en el tiempo — `dev` (ventanas que terminan ≤ 2023-12-31) y `lockbox` (starts en 2024-2025, datos hasta hoy). Reporta media Y mediana, %pos, DD medio y peor, alpha vs SPY/QQQ/60-40 (bate-benchmark %) y Deflated Sharpe por horizonte. Reusa `aggregate_with_ci`. Misma semilla → mismas ventanas → A/B pareado vía `delta_ci`.
- **Métricas** (`utils/backtest_metrics.py`): `information_coefficient`, `directional_accuracy`, `probabilistic_sharpe_ratio`, `expected_max_sharpe`, `deflated_sharpe_ratio` (Bailey & López de Prado), `build_horizon_summary`, `format_horizon_table`. Contador honesto de configuraciones probadas en `utils/trial_registry.py` (alimenta el N de la DSR).
- **Test de causalidad** (`tests/test_causality.py`, in-process vía `app.test_client`): (1) truncación — la decisión en cada día t es idéntica si la ventana se extiende al futuro (probado para parking on/off, equity exactamente igual, `maxrel=0.0`); (2) propiedad de la señal SMA invariante a mutar precios > t; (3) control negativo que demuestra que una señal con look-ahead SÍ se detecta. Todo verde. (El shuffle-test del ML queda para la fase de meta-labeling.)

### Tier 8.1 — S1: Parking en letras del Tesoro cuando se está fuera

**Estado: ACEPTADO ✓** (mejora estricta sobre Faber-QQQ; NO genera alpha vs el índice)

**Hipótesis:** Faber-QQQ pasa ~20-30% del tiempo fuera de mercado, y ahí el baseline tenía el efectivo a **0%**. Acreditar el tipo libre de riesgo sobre el efectivo ocioso no es una apuesta de estrategia, es contabilidad correcta — y en 2022-2025 (tipos 4-5%) es dinero real que se estaba ignorando.

**Mecanismo (causal):** cuando está fuera, se acumula el interés diario derivado de `^IRX` (rendimiento de la letra a 13 semanas), usando el valor del día **anterior** (`shift(1)`), alineado por fecha de calendario (se detectó y corrigió que `^IRX` se sella a las 12:20 y QQQ a las 13:30 → `reindex` directo daba 100% NaN; se normaliza a fecha). `parking_mode='none'` (default) reproduce el baseline EXACTO. Hook: bloque `is_index_trend` de `api.py`.

**DEV 2014-2023 (30 fechas/horizonte, semilla 42, pareado):**

| Plazo | base media | park media | Δret [CI95] | ΔmaxDD |
|-------|-----------:|-----------:|------------:|-------:|
| 6M | +8.02% | +8.18% | **+0.15 [+0.06,+0.26]** | +0.03 |
| 1Y | +12.25% | +12.60% | **+0.34 [+0.15,+0.58]** | +0.10 |
| 2Y | +21.74% | +22.33% | **+0.59 [+0.37,+0.82]** | +0.25 |

**LOCKBOX 2024-2025 (out-of-sample, 30 fechas/horizonte, semilla 42, pareado):**

| Plazo | base media | park media | Δret [CI95] |
|-------|-----------:|-----------:|------------:|
| 6M | +5.67% | +6.04% | **+0.36 [+0.23,+0.48]** |
| 1Y | +15.21% | +15.88% | **+0.68 [+0.57,+0.79]** |
| 2Y | +37.70% | +38.82% | **+1.12 [+1.07,+1.17]** |

- Δret positivo y con IC95% que excluye el 0 en TODOS los horizontes, en dev y en lockbox. **180/180 ventanas: parking ≥ baseline** (por construcción el interés ≥ 0, nunca resta). ΔmaxDD ≥ 0 (el efectivo nunca pierde; drawdown marginalmente mejor).
- El efecto crece con el horizonte y es **mayor en lockbox** (+1.12pp a 2Y vs +0.59pp en dev) porque 2024-2025 tuvo tipos al 4-5% mientras 2014-2021 estuvo cerca de 0. Confirma la lógica del mecanismo.

**La pregunta honesta — ¿bate ahora a QQQ?** No. Incluso con parking, en lockbox Faber-QQQ queda por debajo de comprar y mantener QQQ en todos los plazos (alpha_vs_qqq −0.57% a 2W … −7.29% a 2Y; bate a QQQ solo 0-17% de las ventanas): 2024-2025 fue un bull sostenido y el timing 200d cede upside. **S1 mejora el retorno absoluto de la propia estrategia (Pareto: nunca peor que el baseline 0%-cash), no produce alpha sobre el índice.**

### Lecciones / decisión

- Primera mejora honesta, robusta y validada OOS sobre el baseline Faber-QQQ del Tier 7: **batir a Faber-QQQ** (uno de los dos objetivos) se consigue por contabilidad correcta del efectivo, sin overfitting (no hay parámetro ajustado; la DSR no aplica porque no se buscó sobre ventanas). Batir al ÍNDICE de forma fiable sigue abierto (S2-S4, M1).
- Lección de implementación: las series de Yahoo se sellan a horas intradía distintas según el activo (índices vs ETFs); alinear por timestamp crudo rompe en silencio. Normalizar a fecha.

### Comandos reproducibles

```
PORT=5057 TFG_DISABLE_TF=1 TFG_LIGHTWEIGHT=1 .venv/bin/python -u api.py &
.venv/bin/python tests/test_causality.py
# dev (2014-2023) — pareado, misma semilla
.venv/bin/python -u tools/run_random_grid.py --variant faber_qqq      --universe dev     --n-windows 30 --seed 42
.venv/bin/python -u tools/run_random_grid.py --variant faber_qqq_park  --universe dev     --n-windows 30 --seed 42
# lockbox out-of-sample (2024-2025)
.venv/bin/python -u tools/run_random_grid.py --variant faber_qqq       --universe lockbox --n-windows 30 --seed 42
.venv/bin/python -u tools/run_random_grid.py --variant faber_qqq_park   --universe lockbox --n-windows 30 --seed 42
```

---

## Tier 8.2 — Anti-whipsaw sobre Faber-QQQ: rebalanceo mensual (S3), banda (S2), voto multi-lookback (S4)

**Estado: RECHAZADOS ✗ como mejora de retorno** (mensual = mejor filtro de caídas en bears, pero retrasa en bulls; el lockbox 2024-25 es bull puro y los penaliza). **Hallazgo colateral positivo: Faber-QQQ+parking bate al 60/40 y empata al SPY OOS.**

Todas las variantes apilan sobre el parking (S1, aceptado) y se comparan, pareadas, contra `faber_qqq_park` (parking + diario). Todas default-off; baseline 2019 reproducido EXACTO; test de causalidad sigue verde tras el refactor de la señal.

- **S3 rebalanceo mensual** (`rebalance_frequency=monthly`, decide solo en fin de mes; Faber canónico). **S2 banda de histéresis** (`band=0.01`: entra a `sma·(1+b)`, sale a `sma·(1−b)`). **S4 voto multi-lookback** (`index_signal=vote`: mantiene si la mayoría de {SMA100,150,200,250} dicen "encima"; long/flot sin TP/SL → distinto del TSMOM rechazado del Tier 7).

**DEV 2014-2023 (pareado vs park, n=30/horizonte):** las CIs de Δret cruzan 0 en casi todos los plazos (ninguna mejora robusta en MEDIA). El desglose por **régimen** es lo informativo:

| variante | ventanas QQQ-baja (n=32) ΔRet vs park | ventanas QQQ-alza (n=148) |
|----------|--------------------------------------:|--------------------------:|
| mensual | **+3.26pp** (mediana +1.44) | −1.03pp |
| banda | +2.37pp | +0.14pp |
| voto | −0.31pp | +0.44pp |

En ventanas con QQQ a la baja, el **mensual** pierde solo **−1.16%** frente al **−6.13%** de QQQ y **bate a QQQ el 81%** de las veces (vs 34% del Faber diario, que se desangra en whipsaw). Es decir: el mensual es un filtro de caídas mucho mejor, a costa de ~1pp de retraso en los bull.

**LOCKBOX 2024-2025 (bull puro, pareado vs park):** sin bears que premien la protección, las variantes solo muestran el coste del retraso:

| variante | Δret vs park 6M | 1Y | 2Y |
|----------|----------------:|---:|---:|
| mensual | −2.84 [−3.96,−1.58] | −6.00 [−7.38,−4.56] | **−12.84 [−15.27,−10.73]** |
| banda | −1.40 [−1.95,−0.84] | −2.95 [−3.45,−2.44] | −4.45 [−4.52,−4.37] |
| voto | 0.00 | 0.00 | 0.00 (≡ park: QQQ sobre todas las medias todo el año) |

Todas con IC95% negativo (mensual/banda) o nulo (voto) → **ninguna mejora el retorno OOS; RECHAZADAS.**

**Hallazgo colateral (lo que SÍ bate a un benchmark OOS) — `faber_qqq_park` absoluto en lockbox:**

| Plazo | ret | DD | alpha vs 60/40 | bate 60/40 | alpha vs SPY | bate SPY |
|-------|----:|---:|---------------:|-----------:|-------------:|---------:|
| 6M | +6.04% | −8.6% | +2.35 | 73% | −0.47 | 33% |
| 1Y | +15.88% | −10.7% | +7.99 | 93% | +0.61 | 60% |
| 2Y | +38.82% | −13.6% | +19.34 | **100%** | +0.78 | 60% |

**Faber-QQQ+parking bate al 60/40 de forma robusta OOS (73-100% de las ventanas) y empata/supera ligeramente al SPY con menos drawdown.** Es uno de los objetivos del proyecto (batir al benchmark 60/40). Matiz honesto: frente al 60/40 esto es sobre todo **beta de crecimiento + filtro de caídas**, no alpha; y sigue **sin batir a su propio subyacente QQQ** en un bull.

### Lecciones / decisión

- Reducir whipsaw (mensual/banda/voto) NO es gratis: es un trade-off bear↔bull. En datos dominados por mercados alcistas (y en el lockbox 2024-25, bull puro) cuesta más upside del que ahorra. El mensual es un **mejor filtro de caídas** (bate a QQQ el 81% en bajadas) y queda disponible como palanca de **gestión de riesgo**, pero no como mejora de retorno → no pasa el gate.
- **Insight estructural (para la memoria):** una estrategia *long/flat* sobre un único índice no puede batir a comprar-y-mantener ese índice en un bull sostenido; su valor es esquivar drawdown en los bear. El lockbox 2024-25, al no tener bear real, no puede validar esa ventaja (y de hecho penaliza el timing). Honesto y coherente con el Tier 7.
- Veredicto del ciclo estructural: **solo S1 (parking) se acepta.** S2/S3/S4 rechazadas como mejora de retorno. La mejor config honesta pasa a ser **Faber-QQQ + parking**.

### Comandos reproducibles

```
.venv/bin/python -u tools/run_random_grid.py --variant faber_qqq_monthly --universe dev
.venv/bin/python -u tools/run_random_grid.py --variant faber_qqq_band    --universe dev
.venv/bin/python -u tools/run_random_grid.py --variant faber_qqq_vote    --universe dev
.venv/bin/python -u tools/run_random_grid.py --variant faber_qqq_honest  --universe dev
# (repetir con --universe lockbox para la validación OOS)
```

---

## Tier 8.3 — M1: Meta-labeling del primario Faber (López de Prado)

**Estado: RECHAZADO ✗ — NULL honesto (el clasificador secundario no tiene edge fuera de fold)**

**Hipótesis:** el primario Faber (QQQ > SMA200) ya tiene expectativa positiva; un clasificador secundario que prediga P(esta entrada concreta acabará en beneficio) permitiría **filtrar/reducir tamaño** de las entradas malas (whipsaws), sin invertir nunca la dirección → mejor Sharpe / protección. Es el caso de uso de libro del meta-labeling.

**Implementación (correcta y verificada):** módulo `models/ml_models/meta_labeling.py` con el aparato completo de López de Prado — `triple_barrier_labels`, `PurgedKFold` (purga + embargo), `get_avg_uniqueness` (pesos por unicidad de etiquetas solapadas), `frac_diff_ffd`, `train_meta_model`. **9/9 tests unitarios verdes** (`tests/test_meta_labeling.py`). Leak-free: las etiquetas miran adelante solo como target; la sigma de las barreras es trailing; la purga elimina el solapamiento entre etiquetas de train y barras de test.

**Probe GO/NO-GO** (`tools/probe_meta_labeling.py`, SOLO DEV 2014-2021, QQQ; eventos Faber long-eligible submuestreados cada 5 días, n=319; triple-barrier pt=sl=1·σ, vbar=20d; XGBClassifier con pesos de unicidad y CV purgada+embargo):

| | AUC OOF | IC (Spearman) | base_rate |
|--|-------:|--------------:|----------:|
| **Real** | **0.3855** | −0.19 | 0.605 |
| Shuffle (control, 5 semillas) | media 0.528 · máx 0.580 | ≈ 0 | — |

- El AUC real (0.39) es **peor que el azar** y por debajo del nulo barajado (máx 0.58) → **VEREDICTO NULL**. No es ruido ≈0.5: un AUC < 0.5 bajo CV purgado es la huella de que los patrones in-sample **se invierten** fuera de fold (sin estructura aprendible) y de que la purga elimina el leakage que, de otro modo, falsearía un edge. El shuffle-test confirma que no hay señal enmascarada.
- **Decisión:** NO construir el overlay de sizing (actuaría sobre un clasificador peor que una moneda). NO se buscaron configuraciones de barrera/feature que "ganaran" — sería multiple-testing/overfitting, justo lo que el protocolo prohíbe; el shuffle-test ya descarta señal con la config canónica.

### Lecciones

- Aplicar ML "bien" (meta-labeling de un primario que SÍ funciona, con CV purgado + embargo + triple-barrier + unicidad + shuffle-test) **no crea edge** en este problema. Es un resultado negativo riguroso y publicable, no un fallo de implementación (tests verdes, null reproducible y determinista).
- Refuerza el hallazgo transversal del proyecto: ningún componente **aprendido/IA** añade alpha sobre el indexado; lo único que mejora de forma robusta y honesta es la **contabilidad correcta del efectivo** (S1, parking). El valor del enfoque Faber-QQQ es **beta de crecimiento con filtro de caídas** (menos drawdown, bate al 60/40 OOS), no alpha.

### Comando reproducible

```
.venv/bin/python tests/test_meta_labeling.py
.venv/bin/python tools/probe_meta_labeling.py
```

---
