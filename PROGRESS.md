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
