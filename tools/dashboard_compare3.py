"""
Tier 4.2 — Dashboard 3-way: MED vs AGGR (Tier 4.1) vs AGGR_PLUS (Tier 4.2).

Lee los 3 JSONs:
  /tmp/perf_grid_result.json           — MED baseline
  /tmp/perf_grid_aggr_result.json      — AGGR_KELLY_PCT (Tier 4.1)
  /tmp/perf_grid_aggr_plus_result.json — AGGR_PLUS_C3 (Tier 4.2: AGGR + top-N rotation con ETFs)

Genera HTML standalone con Plotly inline (CDN). Secciones:
  1. KPIs comparados 3 columnas
  2. Resumen de candidatos C1-C5 con verdicts
  3. Tabla por plazo (3 series: ret, Sharpe, %pos)
  4. Bar chart agrupado por plazo
  5. Scatter bot vs B&H (3 series distinguibles)
  6. Tabla pareada de los 40 runs

Uso:
    .venv/bin/python -u tools/run_performance_grid.py             # MED
    .venv/bin/python -u tools/run_performance_grid.py --variant aggr
    .venv/bin/python -u tools/run_performance_grid.py --variant aggr_plus
    .venv/bin/python -u tools/dashboard_compare3.py
    open /tmp/dashboard_compare3.html
"""
import json
import os
import sys
from datetime import datetime
from html import escape

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

PATHS = {
    'MED':       '/tmp/perf_grid_result.json',
    'AGGR':      '/tmp/perf_grid_aggr_result.json',
    'AGGR_PLUS': '/tmp/perf_grid_aggr_plus_result.json',
}
COLORS = {'MED': '#94a3b8', 'AGGR': '#0ea5e9', 'AGGR_PLUS': '#16a34a'}
OUT_PATH = '/tmp/dashboard_compare3.html'


def color_by_sign(x: float) -> str:
    return '#16a34a' if x > 0 else ('#dc2626' if x < 0 else '#64748b')


def stats(runs):
    n = len(runs)
    if n == 0: return None
    return {
        'n':         n,
        'avg_ret':   sum(r['return_pct']  for r in runs) / n,
        'med_ret':   sorted(r['return_pct'] for r in runs)[n//2],
        'avg_alpha': sum(r['alpha_pct']   for r in runs) / n,
        'avg_dd':    sum(r['maxdd_pct']   for r in runs) / n,
        'avg_sh':    sum(r['sharpe']      for r in runs) / n,
        'avg_wr':    sum(r['win_rate_pct']for r in runs) / n,
        'avg_trades':sum(r['total_trades']for r in runs) / n,
        'n_pos':     sum(1 for r in runs if r['return_pct'] > 0),
        'n_alpha':   sum(1 for r in runs if r['alpha_pct'] > 0),
        'n_no_tr':   sum(1 for r in runs if r['total_trades'] == 0),
    }


def main():
    data = {}
    for k, p in PATHS.items():
        if not os.path.exists(p):
            print(f"ERROR: {p} no existe. Ejecuta tools/run_performance_grid.py --variant {k.lower()}")
            sys.exit(1)
        data[k] = json.load(open(p))
    runs = {k: data[k]['runs'] for k in PATHS}
    s = {k: stats(runs[k]) for k in PATHS}

    # ── KPIs 3 columnas ────────────────────────────────────────────────
    kpi_rows = []
    for label, key, fmt in [
        ('avg return (todos los plazos)', 'avg_ret',   '%+.2f%%'),
        ('mediana return',                'med_ret',   '%+.2f%%'),
        ('avg Sharpe',                    'avg_sh',    '%+.2f'),
        ('avg MaxDD',                     'avg_dd',    '%+.2f%%'),
        ('runs positivos',                'n_pos',     '%d/40'),
        ('runs que baten B&H',            'n_alpha',   '%d/40'),
        ('runs sin trades',               'n_no_tr',   '%d/40'),
        ('avg trades/run',                'avg_trades','%.1f'),
        ('avg win_rate',                  'avg_wr',    '%.1f%%'),
    ]:
        m_v, a_v, p_v = s['MED'][key], s['AGGR'][key], s['AGGR_PLUS'][key]
        d_aggr = a_v - m_v
        d_plus = p_v - m_v
        kpi_rows.append(
            f"<tr><td><b>{escape(label)}</b></td>"
            f"<td>{fmt % m_v}</td>"
            f"<td>{fmt % a_v}<span class='delta'>({fmt % d_aggr})</span></td>"
            f"<td>{fmt % p_v}<span class='delta'>({fmt % d_plus})</span></td></tr>"
        )

    # ── Por plazo ──────────────────────────────────────────────────────
    horizons = [h['horizon'] for h in data['MED']['summary_by_horizon']]
    by_h = {k: {h['horizon']: h for h in data[k]['summary_by_horizon']} for k in PATHS}
    horizon_rows = []
    for h in horizons:
        row = f"<tr><td><b>{h}</b></td>"
        for k in ['MED', 'AGGR', 'AGGR_PLUS']:
            row += f"<td style='color:{color_by_sign(by_h[k][h]['avg_return'])}'>{by_h[k][h]['avg_return']:+.2f}%</td>"
        for k in ['MED', 'AGGR', 'AGGR_PLUS']:
            row += f"<td>{by_h[k][h]['avg_sharpe']:+.2f}</td>"
        for k in ['MED', 'AGGR', 'AGGR_PLUS']:
            row += f"<td>{by_h[k][h]['pct_positive']:.0f}%</td>"
        row += "</tr>"
        horizon_rows.append(row)

    # ── Bar chart agrupado por plazo (3 series) ────────────────────────
    bar_fig = make_subplots(rows=1, cols=2, subplot_titles=(
        'avg_return por plazo (3 variantes)', 'avg_Sharpe por plazo'
    ))
    for k in ['MED', 'AGGR', 'AGGR_PLUS']:
        bar_fig.add_trace(go.Bar(
            x=horizons, y=[by_h[k][h]['avg_return'] for h in horizons],
            name=k, marker_color=COLORS[k],
            text=[f"{by_h[k][h]['avg_return']:+.0f}%" for h in horizons], textposition='outside',
        ), row=1, col=1)
        bar_fig.add_trace(go.Bar(
            x=horizons, y=[by_h[k][h]['avg_sharpe'] for h in horizons],
            name=k, marker_color=COLORS[k], showlegend=False,
            text=[f"{by_h[k][h]['avg_sharpe']:+.2f}" for h in horizons], textposition='outside',
        ), row=1, col=2)
    bar_fig.update_layout(barmode='group', height=420, template='plotly_white',
                          margin=dict(l=40, r=20, t=50, b=40))
    bar_html = pio.to_html(bar_fig, include_plotlyjs='cdn', full_html=False, div_id='bar3')

    # ── Scatter bot vs B&H 3 series ────────────────────────────────────
    sc_fig = go.Figure()
    for k in ['MED', 'AGGR', 'AGGR_PLUS']:
        sc_fig.add_trace(go.Scatter(
            x=[r['bh_return_pct'] for r in runs[k]],
            y=[r['return_pct']    for r in runs[k]],
            mode='markers', name=k,
            marker=dict(size=10, opacity=0.6, color=COLORS[k]),
            text=[f"{r['start_date']} ({r['horizon']})" for r in runs[k]],
            hovertemplate='%{text}<br>B&H=%{x:+.1f}%<br>bot=%{y:+.1f}%<extra>'+k+'</extra>',
        ))
    all_xs = [r['bh_return_pct'] for k in PATHS for r in runs[k]]
    all_ys = [r['return_pct']    for k in PATHS for r in runs[k]]
    rng = max(abs(min(all_xs)), abs(max(all_xs)),
              abs(min(all_ys)), abs(max(all_ys))) * 1.05
    sc_fig.add_trace(go.Scatter(
        x=[-rng, rng], y=[-rng, rng], mode='lines',
        line=dict(dash='dash', color='#9ca3af'), name='bot = B&H',
    ))
    sc_fig.add_hline(y=0, line=dict(color='#9ca3af', width=1))
    sc_fig.add_vline(x=0, line=dict(color='#9ca3af', width=1))
    sc_fig.update_layout(
        title='Retorno bot vs Buy & Hold — 3 variantes',
        xaxis_title='Retorno B&H (%)', yaxis_title='Retorno bot (%)',
        height=520, template='plotly_white', margin=dict(l=60, r=20, t=60, b=50),
    )
    sc_html = pio.to_html(sc_fig, include_plotlyjs=False, full_html=False, div_id='sc3')

    # ── Tabla pareada ──────────────────────────────────────────────────
    runs_by_key = {(r['start_date'], r['horizon']): r for r in runs['MED']}
    aggr_by_key = {(r['start_date'], r['horizon']): r for r in runs['AGGR']}
    plus_by_key = {(r['start_date'], r['horizon']): r for r in runs['AGGR_PLUS']}
    pair_rows = []
    for key in sorted(runs_by_key, key=lambda k: (k[0], int(k[1].rstrip('MY')) * (12 if k[1].endswith('M') else 1))):
        m_r = runs_by_key[key]
        a_r = aggr_by_key.get(key)
        p_r = plus_by_key.get(key)
        if not (a_r and p_r): continue
        pair_rows.append(
            f"<tr>"
            f"<td>{m_r['start_date']}</td><td>{m_r['horizon']}</td>"
            f"<td style='color:{color_by_sign(m_r['return_pct'])}'>{m_r['return_pct']:+.2f}%</td>"
            f"<td style='color:{color_by_sign(a_r['return_pct'])}'>{a_r['return_pct']:+.2f}%</td>"
            f"<td style='color:{color_by_sign(p_r['return_pct'])};font-weight:600'>{p_r['return_pct']:+.2f}%</td>"
            f"<td>{m_r['bh_return_pct']:+.2f}%</td>"
            f"<td>{m_r['maxdd_pct']:+.2f}%</td>"
            f"<td>{p_r['maxdd_pct']:+.2f}%</td>"
            f"<td>{m_r['total_trades']}</td>"
            f"<td>{p_r['total_trades']}</td>"
            f"</tr>"
        )

    # ── Verdicts table ─────────────────────────────────────────────────
    verdicts_html = """
    <table>
      <thead><tr><th>Cand</th><th>Descripción</th><th>Δ_Sharpe</th><th>Δ_return</th><th>Δ_MaxDD</th><th>Veredicto</th></tr></thead>
      <tbody>
        <tr><td>C1</td><td>+5 ETFs (XLE/XLF/GLD/MTUM/IWM)</td>
          <td>+0.148 [+0.01,+0.30]</td><td>−5.58pp</td><td>−5.79pp</td>
          <td style="color:#dc2626"><b>RECHAZADO</b><br><small>diluye en bull tech</small></td></tr>
        <tr><td>C2</td><td>Vol-targeting overlay (Moreira-Muir)</td>
          <td>−0.323 sig</td><td>−21.21pp sig</td><td>+10.19pp ✓</td>
          <td style="color:#dc2626"><b>RECHAZADO HARD</b><br><small>corta upside masivo</small></td></tr>
        <tr><td><b>C3</b></td><td><b>Top-N rotation por momentum 60d (con ETFs como pool)</b></td>
          <td>+0.181 [+0.12,+0.25]</td><td><b>+35.34pp [+9.85,+62.99]</b></td><td>−6.19pp</td>
          <td style="color:#16a34a"><b>ACEPTADO ✓</b><br><small>24 ventanas</small></td></tr>
        <tr><td>C4</td><td>Volatility filter (skip si vol fuera [10%,150%])</td>
          <td>−0.035</td><td>−7.09pp</td><td>−0.63pp</td>
          <td style="color:#dc2626"><b>RECHAZADO</b><br><small>filtro casi inactivo</small></td></tr>
        <tr><td>C5</td><td>MR combo (RSI&lt;30 cuando ADX&lt;18)</td>
          <td>0</td><td>0</td><td>0</td>
          <td style="color:#64748b"><b>SIN EFECTO</b><br><small>filtro ADX no se activa</small></td></tr>
      </tbody>
    </table>
    """

    generated_at = datetime.now().isoformat(timespec='seconds')
    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Bot 3-way Comparison · Tier 4.2</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         margin:0; padding:0; background:#f8fafc; color:#0f172a; }}
  header {{ background:#0f172a; color:#f8fafc; padding:24px 32px; }}
  header h1 {{ margin:0 0 6px; font-size:22px; font-weight:600; }}
  header .meta {{ color:#94a3b8; font-size:12px; line-height:1.6; }}
  main {{ padding:24px 32px; max-width:1500px; margin:0 auto; }}
  section {{ background:#fff; padding:18px 22px; margin-bottom:22px;
             border-radius:10px; box-shadow:0 1px 3px rgba(0,0,0,0.06); }}
  section h2 {{ margin:0 0 12px; font-size:16px; color:#0f172a;
                border-bottom:1px solid #e2e8f0; padding-bottom:8px; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th, td {{ padding:8px 10px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  th:first-child, td:first-child {{ text-align:left; }}
  th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
  thead th {{ background:#f1f5f9; font-weight:600; color:#475569;
              border-bottom:2px solid #cbd5e1; cursor:pointer; user-select:none; }}
  tbody tr:hover {{ background:#f8fafc; }}
  .delta {{ display:inline-block; margin-left:8px; font-size:11px; color:#64748b; }}
  .verdict {{ background:#dcfce7; border:1px solid #86efac; padding:14px 18px;
              border-radius:8px; font-size:13px; line-height:1.6; }}
  code {{ background:#f1f5f9; padding:2px 6px; border-radius:4px; font-size:11px; }}
</style>
</head>
<body>
<header>
  <h1>Bot Performance · 3-way comparison · Tier 4.1 + Tier 4.2</h1>
  <div class="meta">
    Generado: {generated_at} · 40 backtests cada variante · 10 fechas × 4 plazos sobre 2018-2024 · sin LLM<br>
    <b>MED</b>: <code>kelly=0.22, signal_pct=0.82</code> · <b>AGGR</b>: <code>kelly=0.30, signal_pct=0.70</code> ·
    <b>AGGR_PLUS</b>: <code>AGGR + top-N rotation 5/15 por momentum 60d (incl. ETFs como pool)</code>
  </div>
</header>
<main>

<section>
  <h2>1 · Veredicto Tier 4.2 — qué pasó cada candidato</h2>
  <div class="verdict">
    <b>De 5 candidatos investigados (Reddit + GitHub + papers), solo C3 pasó el gate Agresivo.</b>
    El bot ya estaba muy bien controlado: vol-targeting (C2) cortó upside,
    ETFs solos (C1) dilutaron en bull tech, vol-filter (C4) y MR combo (C5)
    no se activaron sobre AGGR. Top-N rotation (C3) aprovecha los ETFs como
    pool extendido pero ranqueando por momentum 60d — concentra capital
    donde la convicción es máxima.
  </div>
  {verdicts_html}
</section>

<section>
  <h2>2 · KPIs comparados (3 variantes lado a lado)</h2>
  <table>
    <thead><tr><th>KPI</th><th>MED (baseline)</th><th>AGGR (Tier 4.1)</th><th>AGGR_PLUS (Tier 4.2)</th></tr></thead>
    <tbody>{''.join(kpi_rows)}</tbody>
  </table>
</section>

<section>
  <h2>3 · Resumen por plazo</h2>
  <table>
    <thead><tr>
      <th rowspan="2">plazo</th>
      <th colspan="3">avg_return</th>
      <th colspan="3">avg_Sharpe</th>
      <th colspan="3">%pos</th>
    </tr><tr>
      <th>MED</th><th>AGGR</th><th>PLUS</th>
      <th>MED</th><th>AGGR</th><th>PLUS</th>
      <th>MED</th><th>AGGR</th><th>PLUS</th>
    </tr></thead>
    <tbody>{''.join(horizon_rows)}</tbody>
  </table>
</section>

<section>
  <h2>4 · Bar charts agrupados por plazo</h2>
  {bar_html}
</section>

<section>
  <h2>5 · Bot vs Buy &amp; Hold (3 variantes superpuestas)</h2>
  {sc_html}
</section>

<section>
  <h2>6 · Detalle pareado de los 40 runs (clic en cabecera para ordenar)</h2>
  <table id="detail">
    <thead><tr>
      <th>start</th><th>plazo</th>
      <th>MED ret</th><th>AGGR ret</th><th>PLUS ret</th>
      <th>B&amp;H</th>
      <th>MED DD</th><th>PLUS DD</th>
      <th>MED trades</th><th>PLUS trades</th>
    </tr></thead>
    <tbody>{''.join(pair_rows)}</tbody>
  </table>
</section>

</main>
<script>
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
    print(f"→ Dashboard 3-way: {OUT_PATH}")
    print(f"  Abre con: open {OUT_PATH}")


if __name__ == '__main__':
    main()
