"""
Genera un dashboard HTML standalone (autocontenido, sin servidor) que
visualiza la rentabilidad del bot a partir de /tmp/perf_grid_result.json.

Plotly inline via CDN. Abre en cualquier navegador.

Secciones:
  1. KPIs globales (cards)
  2. Tabla resumen por plazo
  3. Bar chart: avg return / avg alpha / win rate por plazo
  4. Heatmap: retorno por (fecha_inicio × plazo)
  5. Scatter: retorno bot vs B&H (cada punto = un run)
  6. Tabla detallada de todos los runs (ordenable por columna)

Uso:
    .venv/bin/python -u tools/dashboard_performance.py
    open /tmp/dashboard_perf.html        # macOS
"""
import json
import os
import sys
from datetime import datetime
from html import escape

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

IN_PATH  = '/tmp/perf_grid_result.json'
OUT_PATH = '/tmp/dashboard_perf.html'


def color_by_sign(x: float) -> str:
    return '#16a34a' if x >= 0 else '#dc2626'


def kpi_card(label: str, value: str, sub: str = '', accent: str = '#0ea5e9') -> str:
    return f"""
    <div class="kpi" style="border-left:4px solid {accent}">
      <div class="kpi-label">{escape(label)}</div>
      <div class="kpi-value">{escape(value)}</div>
      <div class="kpi-sub">{escape(sub)}</div>
    </div>
    """


def main():
    if not os.path.exists(IN_PATH):
        print(f"ERROR: no existe {IN_PATH}. Ejecuta primero tools/run_performance_grid.py")
        sys.exit(1)
    data = json.load(open(IN_PATH))
    runs = data['runs']
    summary = data['summary_by_horizon']
    if not runs:
        print("ERROR: sin runs en el JSON")
        sys.exit(1)

    n = len(runs)
    avg_ret    = sum(r['return_pct'] for r in runs) / n
    median_ret = sorted(r['return_pct'] for r in runs)[n // 2]
    avg_alpha  = sum(r['alpha_pct'] for r in runs) / n
    n_pos      = sum(1 for r in runs if r['return_pct'] > 0)
    n_alpha    = sum(1 for r in runs if r['alpha_pct'] > 0)
    avg_dd     = sum(r['maxdd_pct'] for r in runs) / n
    avg_sharpe = sum(r['sharpe'] for r in runs) / n
    avg_trades = sum(r['total_trades'] for r in runs) / n
    avg_wr     = sum(r['win_rate_pct'] for r in runs) / n

    # ── KPIs globales ──────────────────────────────────────────────────
    kpis_html = (
        kpi_card('avg return (todos los plazos)', f"{avg_ret:+.2f} %",
                 f"mediana {median_ret:+.2f} %", color_by_sign(avg_ret)) +
        kpi_card('runs positivos', f"{n_pos}/{n}",
                 f"{n_pos/n*100:.0f} % de los backtests cierran en verde",
                 '#16a34a' if n_pos/n >= 0.5 else '#dc2626') +
        kpi_card('runs que baten a B&H', f"{n_alpha}/{n}",
                 f"alpha medio {avg_alpha:+.1f} pp",
                 '#16a34a' if n_alpha/n >= 0.5 else '#dc2626') +
        kpi_card('avg MaxDD', f"{avg_dd:+.2f} %",
                 f"avg Sharpe {avg_sharpe:+.2f}",
                 '#dc2626' if avg_dd < -20 else '#f59e0b') +
        kpi_card('trades / win-rate', f"{avg_trades:.1f} avg",
                 f"win-rate medio {avg_wr:.1f} %",
                 '#0ea5e9')
    )

    # ── Tabla resumen por plazo ────────────────────────────────────────
    h_rows = ''.join(
        f"<tr><td>{s['horizon']}</td><td>{s['n']}</td>"
        f"<td style='color:{color_by_sign(s['avg_return'])}'>{s['avg_return']:+.2f}%</td>"
        f"<td>{s['median_return']:+.2f}%</td>"
        f"<td style='color:{color_by_sign(s['avg_alpha'])}'>{s['avg_alpha']:+.2f}%</td>"
        f"<td style='color:{color_by_sign(s['avg_maxdd'])}'>{s['avg_maxdd']:+.2f}%</td>"
        f"<td>{s['avg_sharpe']:+.2f}</td>"
        f"<td>{s['avg_win_rate']:.1f}%</td>"
        f"<td>{s['pct_positive']:.0f}%</td></tr>"
        for s in summary
    )

    # ── Bar chart por plazo ────────────────────────────────────────────
    horizons = [s['horizon'] for s in summary]
    bar_fig = make_subplots(rows=1, cols=3, subplot_titles=(
        'avg_return (%)', 'avg_alpha vs B&H (pp)', '%pos / win-rate'
    ))
    bar_fig.add_trace(go.Bar(
        x=horizons, y=[s['avg_return'] for s in summary],
        marker_color=[color_by_sign(s['avg_return']) for s in summary],
        text=[f"{s['avg_return']:+.1f}%" for s in summary], textposition='outside',
        name='avg_return', showlegend=False,
    ), row=1, col=1)
    bar_fig.add_trace(go.Bar(
        x=horizons, y=[s['avg_alpha'] for s in summary],
        marker_color=[color_by_sign(s['avg_alpha']) for s in summary],
        text=[f"{s['avg_alpha']:+.1f}pp" for s in summary], textposition='outside',
        name='avg_alpha', showlegend=False,
    ), row=1, col=2)
    bar_fig.add_trace(go.Bar(
        x=horizons, y=[s['pct_positive'] for s in summary],
        marker_color='#0ea5e9', name='%pos',
        text=[f"{s['pct_positive']:.0f}%" for s in summary], textposition='outside',
    ), row=1, col=3)
    bar_fig.add_trace(go.Bar(
        x=horizons, y=[s['avg_win_rate'] for s in summary],
        marker_color='#a78bfa', name='win-rate',
        text=[f"{s['avg_win_rate']:.0f}%" for s in summary], textposition='outside',
    ), row=1, col=3)
    bar_fig.update_layout(
        height=380, margin=dict(l=40, r=20, t=50, b=40),
        barmode='group', template='plotly_white',
    )
    bar_html = pio.to_html(bar_fig, include_plotlyjs='cdn', full_html=False, div_id='bar-by-horizon')

    # ── Heatmap (start_date × plazo) ───────────────────────────────────
    start_dates = sorted({r['start_date'] for r in runs})
    horizon_labels = [h[0] for h in data['horizons']]
    z_ret = []
    z_alpha = []
    text_ret = []
    text_alpha = []
    for sd in start_dates:
        row_ret, row_alpha, txt_r, txt_a = [], [], [], []
        for h_label in horizon_labels:
            cell = next((r for r in runs if r['start_date'] == sd and r['horizon'] == h_label), None)
            if cell is None:
                row_ret.append(None); row_alpha.append(None)
                txt_r.append(''); txt_a.append('')
            else:
                row_ret.append(cell['return_pct']); row_alpha.append(cell['alpha_pct'])
                txt_r.append(f"{cell['return_pct']:+.1f}%")
                txt_a.append(f"{cell['alpha_pct']:+.1f}pp")
        z_ret.append(row_ret); z_alpha.append(row_alpha)
        text_ret.append(txt_r); text_alpha.append(txt_a)

    hm_fig = make_subplots(rows=1, cols=2, subplot_titles=(
        'Retorno bot por (fecha × plazo)', 'Alpha vs B&H por (fecha × plazo)'
    ), horizontal_spacing=0.13)
    hm_fig.add_trace(go.Heatmap(
        z=z_ret, x=horizon_labels, y=start_dates, text=text_ret,
        texttemplate='%{text}', colorscale='RdYlGn', zmid=0,
        colorbar=dict(title='ret %', x=0.42, len=0.85),
        hovertemplate='start=%{y}<br>plazo=%{x}<br>ret=%{z:+.2f}%<extra></extra>',
    ), row=1, col=1)
    hm_fig.add_trace(go.Heatmap(
        z=z_alpha, x=horizon_labels, y=start_dates, text=text_alpha,
        texttemplate='%{text}', colorscale='RdYlGn', zmid=0,
        colorbar=dict(title='alpha pp', x=1.0, len=0.85),
        hovertemplate='start=%{y}<br>plazo=%{x}<br>alpha=%{z:+.2f}pp<extra></extra>',
    ), row=1, col=2)
    hm_fig.update_layout(
        height=480, margin=dict(l=80, r=20, t=50, b=60), template='plotly_white',
    )
    hm_html = pio.to_html(hm_fig, include_plotlyjs=False, full_html=False, div_id='heatmap')

    # ── Scatter bot vs B&H ─────────────────────────────────────────────
    sc_fig = go.Figure()
    for h_label in horizon_labels:
        rs = [r for r in runs if r['horizon'] == h_label]
        if not rs:
            continue
        sc_fig.add_trace(go.Scatter(
            x=[r['bh_return_pct'] for r in rs],
            y=[r['return_pct'] for r in rs],
            mode='markers', name=h_label,
            text=[f"{r['start_date']} ({h_label})" for r in rs],
            marker=dict(size=10, opacity=0.75),
            hovertemplate='%{text}<br>B&H=%{x:+.1f}%<br>bot=%{y:+.1f}%<extra></extra>',
        ))
    # Linea y=x (paridad con B&H)
    rng = max(abs(min(r['bh_return_pct'] for r in runs)),
              abs(max(r['bh_return_pct'] for r in runs)),
              abs(min(r['return_pct']    for r in runs)),
              abs(max(r['return_pct']    for r in runs))) * 1.05
    sc_fig.add_trace(go.Scatter(
        x=[-rng, rng], y=[-rng, rng], mode='lines',
        line=dict(dash='dash', color='#9ca3af'), name='bot = B&H', showlegend=True,
    ))
    sc_fig.add_hline(y=0, line=dict(color='#9ca3af', width=1))
    sc_fig.add_vline(x=0, line=dict(color='#9ca3af', width=1))
    sc_fig.update_layout(
        title='Retorno bot vs Buy & Hold (cada punto = un backtest)',
        xaxis_title='Retorno B&H (%)',
        yaxis_title='Retorno bot (%)',
        height=480, template='plotly_white',
        margin=dict(l=60, r=20, t=60, b=50),
    )
    sc_html = pio.to_html(sc_fig, include_plotlyjs=False, full_html=False, div_id='scatter-vs-bh')

    # ── Tabla detallada ────────────────────────────────────────────────
    rows_sorted = sorted(runs, key=lambda r: (r['start_date'], r['horizon_days']))
    rows_html = ''.join(
        f"<tr>"
        f"<td>{r['start_date']}</td>"
        f"<td>{r['end_date']}</td>"
        f"<td>{r['horizon']}</td>"
        f"<td style='color:{color_by_sign(r['return_pct'])};font-weight:600'>{r['return_pct']:+.2f}%</td>"
        f"<td>{r['bh_return_pct']:+.2f}%</td>"
        f"<td style='color:{color_by_sign(r['alpha_pct'])}'>{r['alpha_pct']:+.2f} pp</td>"
        f"<td style='color:{color_by_sign(r['maxdd_pct'])}'>{r['maxdd_pct']:+.2f}%</td>"
        f"<td>{r['sharpe']:+.2f}</td>"
        f"<td>{r['profit_factor']:.2f}</td>"
        f"<td>{r['win_rate_pct']:.1f}%</td>"
        f"<td>{r['total_trades']}</td>"
        f"</tr>"
        for r in rows_sorted
    )

    # ── HTML envelope ──────────────────────────────────────────────────
    cfg = data.get('config', {})
    cfg_pretty = ', '.join(f"{k}={v}" for k, v in cfg.items())
    generated_at = data.get('generated_at', datetime.now().isoformat(timespec='seconds'))

    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Bot Performance Dashboard</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         margin: 0; padding: 0; background:#f8fafc; color:#0f172a; }}
  header {{ background:#0f172a; color:#f8fafc; padding:24px 32px; }}
  header h1 {{ margin:0 0 6px 0; font-size:22px; font-weight:600; }}
  header .meta {{ color:#94a3b8; font-size:13px; line-height:1.5; }}
  main {{ padding:24px 32px; max-width:1400px; margin:0 auto; }}
  section {{ background:#fff; padding:18px 22px; margin-bottom:22px;
             border-radius:10px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
  section h2 {{ margin:0 0 12px 0; font-size:16px; color:#0f172a;
                border-bottom:1px solid #e2e8f0; padding-bottom:8px; }}
  .kpi-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
               gap:14px; }}
  .kpi {{ background:#f8fafc; padding:14px 16px; border-radius:8px; }}
  .kpi-label {{ font-size:11px; color:#64748b; text-transform:uppercase;
                letter-spacing:0.05em; }}
  .kpi-value {{ font-size:24px; font-weight:700; margin:6px 0 4px 0; }}
  .kpi-sub {{ font-size:12px; color:#64748b; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th, td {{ padding:8px 10px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2),
  th:nth-child(3), td:nth-child(3) {{ text-align:left; }}
  thead th {{ background:#f1f5f9; font-weight:600; color:#475569;
              border-bottom:2px solid #cbd5e1; cursor:pointer; user-select:none; }}
  thead th:hover {{ background:#e2e8f0; }}
  tbody tr:hover {{ background:#f8fafc; }}
  .takeaways {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                 gap:14px; }}
  .takeaway {{ background:#f8fafc; padding:14px 16px; border-radius:8px;
               border-left:3px solid #0ea5e9; font-size:13px; line-height:1.5; }}
  .takeaway b {{ display:block; color:#0f172a; margin-bottom:4px; }}
  code {{ background:#f1f5f9; padding:2px 6px; border-radius:4px; font-size:12px; }}
</style>
</head>
<body>
<header>
  <h1>Bot Performance Dashboard — ¿es rentable?</h1>
  <div class="meta">
    <div>Generado: {generated_at} · Runs: {n} · Variante: <code>MED_NOLLM</code> (los gates Tier 1.2/2.3/3.3 confirmaron que las variantes con LLM son estadísticamente equivalentes)</div>
    <div>Config: <code>{escape(cfg_pretty)}</code></div>
  </div>
</header>
<main>

<section>
  <h2>1 · KPIs globales</h2>
  <div class="kpi-grid">
    {kpis_html}
  </div>
</section>

<section>
  <h2>2 · Lectura honesta (lo que dicen los números)</h2>
  <div class="takeaways">
    <div class="takeaway">
      <b>Plazos cortos pierden</b>
      A 3 meses el avg_return es <code>{summary[0]['avg_return']:+.1f}%</code>
      con sólo <code>{summary[0]['pct_positive']:.0f}%</code> de runs positivos.
      Los costes (slippage/comisiones) y el ruido pesan más que la señal.
    </div>
    <div class="takeaway">
      <b>Plazos largos sí compensan</b>
      A 2 años el avg_return sube a <code>{summary[-1]['avg_return']:+.1f}%</code>
      con <code>{summary[-1]['pct_positive']:.0f}%</code> positivos. El bot
      necesita tiempo para que la trend-following capture múltiples ciclos.
    </div>
    <div class="takeaway">
      <b>Casi nunca bate a Buy&amp;Hold</b>
      Sólo <code>{n_alpha}/{n}</code> ({n_alpha/n*100:.0f}%) runs tienen alpha &gt; 0
      vs B&amp;H. La era 2018-2024 fue estructuralmente alcista (BTC +1500%);
      el bot capa upside con sus filtros.
    </div>
    <div class="takeaway">
      <b>MaxDD controlado</b>
      avg_MaxDD de <code>{avg_dd:+.1f}%</code>. Los filtros (SMA200, ADX, crash_filter,
      vol-scaling) cumplen su función protectora. La promesa del bot es
      menor riesgo, no más retorno.
    </div>
  </div>
</section>

<section>
  <h2>3 · Tabla resumen por plazo</h2>
  <table>
    <thead><tr>
      <th>plazo</th><th>n</th><th>avg_ret</th><th>med_ret</th>
      <th>avg_alpha</th><th>avg_DD</th><th>Sharpe</th>
      <th>win_rate</th><th>%pos</th>
    </tr></thead>
    <tbody>{h_rows}</tbody>
  </table>
</section>

<section>
  <h2>4 · Comparativa por plazo</h2>
  {bar_html}
</section>

<section>
  <h2>5 · Heatmap por (fecha de inicio × plazo)</h2>
  {hm_html}
</section>

<section>
  <h2>6 · Bot vs Buy &amp; Hold (paridad sobre la diagonal)</h2>
  {sc_html}
</section>

<section>
  <h2>7 · Detalle de todos los runs (clic en cabecera para ordenar)</h2>
  <table id="detail-table">
    <thead><tr>
      <th>start</th><th>end</th><th>plazo</th>
      <th>retorno</th><th>B&amp;H</th><th>alpha</th>
      <th>MaxDD</th><th>Sharpe</th><th>PF</th>
      <th>win_rate</th><th>trades</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</section>

</main>
<script>
// Mini sortable table
const table = document.getElementById('detail-table');
const headers = table.querySelectorAll('thead th');
let asc = true; let lastIdx = -1;
const num = s => parseFloat(s.replace(/[^-0-9.]/g, ''));
headers.forEach((th, idx) => {{
  th.addEventListener('click', () => {{
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    asc = (idx === lastIdx) ? !asc : true; lastIdx = idx;
    rows.sort((a, b) => {{
      const av = a.children[idx].textContent;
      const bv = b.children[idx].textContent;
      const an = num(av); const bn = num(bv);
      if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
      return asc ? av.localeCompare(bv) : bv.localeCompare(av);
    }});
    rows.forEach(r => tbody.appendChild(r));
  }});
}});
</script>
</body>
</html>
"""

    with open(OUT_PATH, 'w') as f:
        f.write(html)
    print(f"→ Dashboard generado: {OUT_PATH}")
    print(f"  Abre con: open {OUT_PATH}")


if __name__ == '__main__':
    main()
