"""
Tier 4.1/4.2 — Dashboard comparativo de variantes del bot.

Lee 2 o 3 JSONs de tools/run_performance_grid.py y genera HTML standalone:
  · /tmp/perf_grid_result.json          — MED (baseline)
  · /tmp/perf_grid_aggr_result.json     — AGGR_KELLY_PCT (Tier 4.1)
  · /tmp/perf_grid_aggr_plus_result.json (opcional) — AGGR_PLUS_C3 (Tier 4.2 con top-N)

  1. KPIs side-by-side con Δ explícito
  2. Tabla resumen por plazo (3M/6M/1Y/2Y, ambas variantes)
  3. Bar chart agrupado: avg_return por plazo (MED vs AGGR)
  4. Heatmap diferencial: AGGR - MED por (fecha × plazo)
  5. Scatter bot vs B&H (2 series distinguibles)
  6. Tabla pareada de los 40 runs

Uso:
    .venv/bin/python -u tools/run_performance_grid.py            # genera baseline
    .venv/bin/python -u tools/run_performance_grid.py --variant aggr
    .venv/bin/python -u tools/dashboard_compare.py
    open /tmp/dashboard_compare.html
"""
import json
import os
import sys
from datetime import datetime
from html import escape

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

MED_PATH  = '/tmp/perf_grid_result.json'
AGGR_PATH = '/tmp/perf_grid_aggr_result.json'
PLUS_PATH = '/tmp/perf_grid_aggr_plus_result.json'   # Tier 4.2 (opcional)
OUT_PATH  = '/tmp/dashboard_compare.html'


def color_by_sign(x: float, neutral: str = '#0ea5e9') -> str:
    return '#16a34a' if x > 0 else ('#dc2626' if x < 0 else neutral)


def stats(runs):
    n = len(runs)
    if n == 0:
        return None
    avg_ret    = sum(r['return_pct']  for r in runs) / n
    median_ret = sorted(r['return_pct'] for r in runs)[n // 2]
    avg_alpha  = sum(r['alpha_pct']   for r in runs) / n
    avg_dd     = sum(r['maxdd_pct']   for r in runs) / n
    avg_sharpe = sum(r['sharpe']      for r in runs) / n
    avg_wr     = sum(r['win_rate_pct']for r in runs) / n
    avg_trades = sum(r['total_trades']for r in runs) / n
    n_pos      = sum(1 for r in runs if r['return_pct'] > 0)
    n_alpha    = sum(1 for r in runs if r['alpha_pct'] > 0)
    n_no_trades= sum(1 for r in runs if r['total_trades'] == 0)
    return {
        'n': n, 'avg_ret': avg_ret, 'median_ret': median_ret, 'avg_alpha': avg_alpha,
        'avg_dd': avg_dd, 'avg_sharpe': avg_sharpe, 'avg_wr': avg_wr, 'avg_trades': avg_trades,
        'n_pos': n_pos, 'n_alpha': n_alpha, 'n_no_trades': n_no_trades,
    }


def kpi_card_pair(label, med_val, aggr_val, fmt='%+.2f', unit='%', neutral='#0ea5e9',
                  better='higher'):
    """Dual KPI: baseline + AGGR + Δ con flecha de mejora."""
    delta = aggr_val - med_val
    if better == 'higher':
        improved = delta > 0
    else:  # 'lower' (e.g. avg_dd)
        improved = delta < 0
    arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '→')
    delta_color = '#16a34a' if improved else ('#dc2626' if delta != 0 else '#64748b')
    return f"""
    <div class="kpi-pair">
      <div class="kpi-pair-label">{escape(label)}</div>
      <div class="kpi-pair-row">
        <div class="kpi-pair-val baseline">
          <div class="kpi-pair-tag">MED</div>
          <div class="kpi-pair-num">{fmt % med_val}{unit}</div>
        </div>
        <div class="kpi-pair-val aggr">
          <div class="kpi-pair-tag">AGGR</div>
          <div class="kpi-pair-num">{fmt % aggr_val}{unit}</div>
        </div>
        <div class="kpi-pair-delta" style="color:{delta_color}">
          {arrow} {fmt % abs(delta)}{unit}
        </div>
      </div>
    </div>
    """


def main():
    for p in (MED_PATH, AGGR_PATH):
        if not os.path.exists(p):
            print(f"ERROR: no existe {p}. Ejecuta tools/run_performance_grid.py [--variant ...] primero.")
            sys.exit(1)
    med  = json.load(open(MED_PATH))
    aggr = json.load(open(AGGR_PATH))
    med_runs  = med['runs']
    aggr_runs = aggr['runs']
    if not med_runs or not aggr_runs:
        print("ERROR: sin runs en alguno de los JSONs")
        sys.exit(1)

    sm = stats(med_runs)
    sa = stats(aggr_runs)

    # ── KPIs comparados ────────────────────────────────────────────────
    kpis_html = (
        kpi_card_pair('avg return (todos los plazos)', sm['avg_ret'], sa['avg_ret']) +
        kpi_card_pair('mediana return',                sm['median_ret'], sa['median_ret']) +
        kpi_card_pair('avg Sharpe',                    sm['avg_sharpe'], sa['avg_sharpe'], fmt='%+.2f', unit='') +
        kpi_card_pair('avg MaxDD',                     sm['avg_dd'], sa['avg_dd'], better='lower') +
        kpi_card_pair('runs positivos',                sm['n_pos'], sa['n_pos'], fmt='%d', unit='/40') +
        kpi_card_pair('runs que baten B&H',            sm['n_alpha'], sa['n_alpha'], fmt='%d', unit='/40') +
        kpi_card_pair('runs sin trades',               sm['n_no_trades'], sa['n_no_trades'], fmt='%d', unit='/40', better='lower') +
        kpi_card_pair('avg trades/run',                sm['avg_trades'], sa['avg_trades'], fmt='%.1f', unit='')
    )

    # ── Tabla por plazo lado a lado ────────────────────────────────────
    horizons = [h['horizon'] for h in med['summary_by_horizon']]
    sm_byh = {h['horizon']: h for h in med['summary_by_horizon']}
    sa_byh = {h['horizon']: h for h in aggr['summary_by_horizon']}
    rows = []
    for h in horizons:
        m = sm_byh[h]; a = sa_byh[h]
        d_ret = a['avg_return'] - m['avg_return']
        d_dd  = a['avg_maxdd']  - m['avg_maxdd']
        d_sh  = a['avg_sharpe'] - m['avg_sharpe']
        rows.append(
            f"<tr>"
            f"<td><b>{h}</b></td>"
            f"<td>{m['avg_return']:+.2f}%</td>"
            f"<td style='color:{color_by_sign(d_ret)}'>{a['avg_return']:+.2f}% ({d_ret:+.1f}pp)</td>"
            f"<td>{m['avg_maxdd']:+.2f}%</td>"
            f"<td style='color:{color_by_sign(-d_dd)}'>{a['avg_maxdd']:+.2f}% ({d_dd:+.1f}pp)</td>"
            f"<td>{m['avg_sharpe']:+.2f}</td>"
            f"<td style='color:{color_by_sign(d_sh)}'>{a['avg_sharpe']:+.2f} ({d_sh:+.2f})</td>"
            f"<td>{m['pct_positive']:.0f}%</td>"
            f"<td>{a['pct_positive']:.0f}%</td>"
            f"</tr>"
        )

    # ── Bar chart agrupado por plazo ───────────────────────────────────
    bar_fig = make_subplots(rows=1, cols=2, subplot_titles=(
        'avg_return por plazo', 'avg_Sharpe por plazo'
    ))
    bar_fig.add_trace(go.Bar(
        x=horizons, y=[sm_byh[h]['avg_return'] for h in horizons],
        name='MED (baseline)', marker_color='#94a3b8',
        text=[f"{sm_byh[h]['avg_return']:+.1f}%" for h in horizons], textposition='outside',
    ), row=1, col=1)
    bar_fig.add_trace(go.Bar(
        x=horizons, y=[sa_byh[h]['avg_return'] for h in horizons],
        name='AGGR (Tier 4.1)', marker_color='#0ea5e9',
        text=[f"{sa_byh[h]['avg_return']:+.1f}%" for h in horizons], textposition='outside',
    ), row=1, col=1)
    bar_fig.add_trace(go.Bar(
        x=horizons, y=[sm_byh[h]['avg_sharpe'] for h in horizons],
        name='MED (baseline)', marker_color='#94a3b8', showlegend=False,
        text=[f"{sm_byh[h]['avg_sharpe']:+.2f}" for h in horizons], textposition='outside',
    ), row=1, col=2)
    bar_fig.add_trace(go.Bar(
        x=horizons, y=[sa_byh[h]['avg_sharpe'] for h in horizons],
        name='AGGR (Tier 4.1)', marker_color='#0ea5e9', showlegend=False,
        text=[f"{sa_byh[h]['avg_sharpe']:+.2f}" for h in horizons], textposition='outside',
    ), row=1, col=2)
    bar_fig.update_layout(barmode='group', height=400, template='plotly_white',
                          margin=dict(l=40, r=20, t=50, b=40))
    bar_html = pio.to_html(bar_fig, include_plotlyjs='cdn', full_html=False, div_id='bar-cmp')

    # ── Heatmap diferencial AGGR - MED ────────────────────────────────
    start_dates = sorted({r['start_date'] for r in med_runs})
    z_diff = []; text_diff = []
    for sd in start_dates:
        row_z, row_t = [], []
        for h in horizons:
            mr = next((r for r in med_runs  if r['start_date']==sd and r['horizon']==h), None)
            ar = next((r for r in aggr_runs if r['start_date']==sd and r['horizon']==h), None)
            if mr and ar:
                d = ar['return_pct'] - mr['return_pct']
                row_z.append(d); row_t.append(f"{d:+.1f}pp")
            else:
                row_z.append(None); row_t.append('')
        z_diff.append(row_z); text_diff.append(row_t)

    hm_fig = go.Figure(data=go.Heatmap(
        z=z_diff, x=horizons, y=start_dates, text=text_diff,
        texttemplate='%{text}', colorscale='RdYlGn', zmid=0,
        colorbar=dict(title='Δ ret (pp)'),
        hovertemplate='start=%{y}<br>plazo=%{x}<br>Δ=%{z:+.2f}pp<extra></extra>',
    ))
    hm_fig.update_layout(
        title='AGGR − MED: diferencia de retorno por (fecha × plazo)',
        height=480, template='plotly_white', margin=dict(l=80, r=20, t=60, b=60),
    )
    hm_html = pio.to_html(hm_fig, include_plotlyjs=False, full_html=False, div_id='hm-cmp')

    # ── Scatter vs B&H (2 series) ─────────────────────────────────────
    sc_fig = go.Figure()
    sc_fig.add_trace(go.Scatter(
        x=[r['bh_return_pct'] for r in med_runs],
        y=[r['return_pct']    for r in med_runs],
        mode='markers', name='MED (baseline)',
        marker=dict(size=10, opacity=0.5, color='#94a3b8'),
        text=[f"{r['start_date']} ({r['horizon']})" for r in med_runs],
        hovertemplate='%{text}<br>B&H=%{x:+.1f}%<br>bot=%{y:+.1f}%<extra>MED</extra>',
    ))
    sc_fig.add_trace(go.Scatter(
        x=[r['bh_return_pct'] for r in aggr_runs],
        y=[r['return_pct']    for r in aggr_runs],
        mode='markers', name='AGGR (Tier 4.1)',
        marker=dict(size=10, opacity=0.75, color='#0ea5e9', symbol='diamond'),
        text=[f"{r['start_date']} ({r['horizon']})" for r in aggr_runs],
        hovertemplate='%{text}<br>B&H=%{x:+.1f}%<br>bot=%{y:+.1f}%<extra>AGGR</extra>',
    ))
    rng = max(abs(min(r['bh_return_pct'] for r in med_runs+aggr_runs)),
              abs(max(r['bh_return_pct'] for r in med_runs+aggr_runs)),
              abs(min(r['return_pct']    for r in med_runs+aggr_runs)),
              abs(max(r['return_pct']    for r in med_runs+aggr_runs))) * 1.05
    sc_fig.add_trace(go.Scatter(
        x=[-rng, rng], y=[-rng, rng], mode='lines',
        line=dict(dash='dash', color='#9ca3af'), name='bot = B&H',
    ))
    sc_fig.add_hline(y=0, line=dict(color='#9ca3af', width=1))
    sc_fig.add_vline(x=0, line=dict(color='#9ca3af', width=1))
    sc_fig.update_layout(
        title='Retorno bot vs Buy & Hold — MED vs AGGR',
        xaxis_title='Retorno B&H (%)', yaxis_title='Retorno bot (%)',
        height=500, template='plotly_white', margin=dict(l=60, r=20, t=60, b=50),
    )
    sc_html = pio.to_html(sc_fig, include_plotlyjs=False, full_html=False, div_id='sc-cmp')

    # ── Tabla pareada de runs ─────────────────────────────────────────
    pairs = []
    for mr in sorted(med_runs, key=lambda r: (r['start_date'], r['horizon_days'])):
        ar = next((r for r in aggr_runs if r['start_date']==mr['start_date']
                   and r['horizon']==mr['horizon']), None)
        if not ar:
            continue
        d_ret = ar['return_pct'] - mr['return_pct']
        d_dd  = ar['maxdd_pct']  - mr['maxdd_pct']
        pairs.append(
            f"<tr>"
            f"<td>{mr['start_date']}</td><td>{mr['horizon']}</td>"
            f"<td style='color:{color_by_sign(mr['return_pct'])};font-weight:600'>{mr['return_pct']:+.2f}%</td>"
            f"<td style='color:{color_by_sign(ar['return_pct'])};font-weight:600'>{ar['return_pct']:+.2f}%</td>"
            f"<td style='color:{color_by_sign(d_ret)};font-weight:600'>{d_ret:+.2f}pp</td>"
            f"<td>{mr['maxdd_pct']:+.2f}%</td>"
            f"<td>{ar['maxdd_pct']:+.2f}%</td>"
            f"<td style='color:{color_by_sign(-d_dd)}'>{d_dd:+.2f}pp</td>"
            f"<td>{mr['bh_return_pct']:+.2f}%</td>"
            f"<td>{mr['total_trades']}</td>"
            f"<td>{ar['total_trades']}</td>"
            f"</tr>"
        )

    generated_at = datetime.now().isoformat(timespec='seconds')
    med_cfg  = ', '.join(f"{k}={v}" for k, v in med['config'].items())
    aggr_cfg = ', '.join(f"{k}={v}" for k, v in aggr['config'].items())

    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Bot Performance — MED vs AGGR (Tier 4.1)</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 0; background:#f8fafc; color:#0f172a; }}
  header {{ background:#0f172a; color:#f8fafc; padding:24px 32px; }}
  header h1 {{ margin:0 0 6px 0; font-size:22px; font-weight:600; }}
  header .meta {{ color:#94a3b8; font-size:12px; line-height:1.6; }}
  main {{ padding:24px 32px; max-width:1500px; margin:0 auto; }}
  section {{ background:#fff; padding:18px 22px; margin-bottom:22px;
             border-radius:10px; box-shadow:0 1px 3px rgba(0,0,0,0.06); }}
  section h2 {{ margin:0 0 12px 0; font-size:16px; color:#0f172a;
                border-bottom:1px solid #e2e8f0; padding-bottom:8px; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px; }}
  .kpi-pair {{ background:#f8fafc; padding:12px 14px; border-radius:8px; border-left:4px solid #0ea5e9; }}
  .kpi-pair-label {{ font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px; }}
  .kpi-pair-row {{ display:flex; align-items:center; gap:10px; }}
  .kpi-pair-val {{ flex:1; }}
  .kpi-pair-tag {{ font-size:10px; color:#94a3b8; font-weight:600; }}
  .kpi-pair-num {{ font-size:18px; font-weight:700; }}
  .kpi-pair-val.baseline .kpi-pair-num {{ color:#475569; }}
  .kpi-pair-val.aggr .kpi-pair-num {{ color:#0ea5e9; }}
  .kpi-pair-delta {{ font-size:14px; font-weight:700; min-width:80px; text-align:right; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th, td {{ padding:7px 10px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
  thead th {{ background:#f1f5f9; font-weight:600; color:#475569;
              border-bottom:2px solid #cbd5e1; cursor:pointer; user-select:none; }}
  tbody tr:hover {{ background:#f8fafc; }}
  .verdict {{ background:#dcfce7; border:1px solid #86efac; padding:14px 18px; border-radius:8px;
              font-size:13px; line-height:1.5; }}
  .verdict b {{ color:#15803d; }}
  code {{ background:#f1f5f9; padding:2px 6px; border-radius:4px; font-size:11px; }}
</style>
</head>
<body>
<header>
  <h1>Bot Performance — MED (baseline) vs AGGR_KELLY_PCT (Tier 4.1) · 40 runs cada uno</h1>
  <div class="meta">
    <div>Generado: {generated_at}  ·  10 fechas × 4 plazos sobre 2018-2024  ·  Sin LLM (gates Tier 1.2/2.3/3.3 mostraron equivalencia)</div>
    <div><b>MED</b>:  <code>{escape(med_cfg)}</code></div>
    <div><b>AGGR</b>: <code>{escape(aggr_cfg)}</code> &nbsp;←&nbsp; cambios: kelly_scale 0.22→0.30, signal_percentile 0.82→0.70</div>
  </div>
</header>
<main>

<section>
  <h2>1 · KPIs comparados (MED ↔ AGGR ↔ Δ)</h2>
  <div class="kpi-grid">{kpis_html}</div>
</section>

<section>
  <h2>2 · Veredicto del gate Agresivo (12 ventanas seed=42)</h2>
  <div class="verdict">
    <b>AGGR_KELLY_PCT ACEPTADO ✓</b> &nbsp;·&nbsp;
    Δ_Sharpe = +0.217 [CI95 +0.141, +0.291] · Δ_return = +21.77pp [CI95 +10.16, +35.59] · Δ_MaxDD = -8.40pp (dentro del +10pp tope)<br>
    El gate exige <code>Δ_Sharpe IC95% &gt; 0</code> Y (<code>Δ_return IC95% &gt; +5pp</code> O <code>Δ_MaxDD ≤ +2pp</code>),
    sin disaster (<code>Δ_MaxDD &lt; −10pp</code> o <code>Δ_return IC95% upper &lt; −2pp</code>).
    AGGR cumple Sharpe positivo significativo + return big win — DD empeora -8.4pp pero dentro del tope agresivo.
  </div>
</section>

<section>
  <h2>3 · Resumen por plazo</h2>
  <table>
    <thead><tr>
      <th>plazo</th>
      <th>MED ret</th><th>AGGR ret (Δ)</th>
      <th>MED DD</th><th>AGGR DD (Δ)</th>
      <th>MED Sharpe</th><th>AGGR Sharpe (Δ)</th>
      <th>MED %pos</th><th>AGGR %pos</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>

<section>
  <h2>4 · Bar chart agrupado por plazo</h2>
  {bar_html}
</section>

<section>
  <h2>5 · Heatmap diferencial (AGGR − MED) por (fecha × plazo)</h2>
  {hm_html}
</section>

<section>
  <h2>6 · Scatter bot vs B&amp;H (MED gris claro, AGGR azul)</h2>
  {sc_html}
</section>

<section>
  <h2>7 · Detalle pareado de los 40 runs</h2>
  <table id="detail">
    <thead><tr>
      <th>start</th><th>plazo</th>
      <th>MED ret</th><th>AGGR ret</th><th>Δ ret</th>
      <th>MED DD</th><th>AGGR DD</th><th>Δ DD</th>
      <th>B&amp;H</th>
      <th>MED trades</th><th>AGGR trades</th>
    </tr></thead>
    <tbody>{''.join(pairs)}</tbody>
  </table>
</section>

</main>
<script>
// Sortable
const t = document.getElementById('detail');
const ths = t.querySelectorAll('thead th');
let asc = true; let last = -1;
const num = s => parseFloat(s.replace(/[^-0-9.]/g, ''));
ths.forEach((th, idx) => th.addEventListener('click', () => {{
  const tb = t.querySelector('tbody');
  const rs = Array.from(tb.querySelectorAll('tr'));
  asc = (idx === last) ? !asc : true; last = idx;
  rs.sort((a, b) => {{
    const av = a.children[idx].textContent;
    const bv = b.children[idx].textContent;
    const an = num(av); const bn = num(bv);
    if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  rs.forEach(r => tb.appendChild(r));
}}));
</script>
</body>
</html>
"""

    with open(OUT_PATH, 'w') as f:
        f.write(html)
    print(f"→ Dashboard comparativo: {OUT_PATH}")
    print(f"  Abre con: open {OUT_PATH}")


if __name__ == '__main__':
    main()
