"""
Tier 5 — Dashboard 4-way: MED vs AGGR_PLUS_FAMOUS vs SWING_FAMOUS vs SWING_BROAD.

Lee 4 JSONs producidos por tools/run_performance_grid.py (variantes med,
aggr_plus, swing, swing_broad) y compara performance lado a lado:

  1. KPIs 4 columnas con Δ entre variantes
  2. Tabla por plazo (incluye 1W/2W/1M/2M para variantes swing)
  3. Bar chart agrupado por plazo
  4. Heatmap de retornos (start_date × plazo) por variante
  5. Bankruptcy guardrail: cuántas ventanas con DD < −40%

Uso:
    .venv/bin/python -u tools/run_performance_grid.py --variant med
    .venv/bin/python -u tools/run_performance_grid.py --variant aggr_plus
    .venv/bin/python -u tools/run_performance_grid.py --variant swing
    .venv/bin/python -u tools/run_performance_grid.py --variant swing_broad
    .venv/bin/python -u tools/dashboard_compare4.py
    open /tmp/dashboard_compare4.html
"""
import json
import os
import sys
from datetime import datetime
from html import escape

import plotly.graph_objects as go
import plotly.io as pio

PATHS = {
    'MED':              '/tmp/perf_grid_result.json',
    'AGGR_PLUS':        '/tmp/perf_grid_aggr_plus_result.json',
    'SWING':            '/tmp/perf_grid_swing_result.json',
    'SWING_BROAD':      '/tmp/perf_grid_swing_broad_result.json',
}
COLORS = {
    'MED':         '#94a3b8',
    'AGGR_PLUS':   '#16a34a',
    'SWING':       '#0ea5e9',
    'SWING_BROAD': '#a21caf',
}
OUT_PATH = '/tmp/dashboard_compare4.html'

DD_DISASTER_THRESHOLD = -40.0  # Tier 5 bankruptcy guardrail


def color_by_sign(x: float) -> str:
    return '#16a34a' if x > 0 else ('#dc2626' if x < 0 else '#64748b')


def stats(runs):
    n = len(runs)
    if n == 0:
        return None
    return {
        'n':            n,
        'avg_ret':      sum(r['return_pct']  for r in runs) / n,
        'med_ret':      sorted(r['return_pct'] for r in runs)[n // 2],
        'avg_alpha':    sum(r['alpha_pct']   for r in runs) / n,
        'avg_dd':       sum(r['maxdd_pct']   for r in runs) / n,
        'worst_dd':     min(r['maxdd_pct']   for r in runs),
        'avg_sh':       sum(r['sharpe']      for r in runs) / n,
        'avg_wr':       sum(r['win_rate_pct']for r in runs) / n,
        'avg_trades':   sum(r['total_trades']for r in runs) / n,
        'n_pos':        sum(1 for r in runs if r['return_pct'] > 0),
        'n_alpha':      sum(1 for r in runs if r['alpha_pct'] > 0),
        'n_no_tr':      sum(1 for r in runs if r['total_trades'] == 0),
        'n_disaster':   sum(1 for r in runs if r['maxdd_pct'] < DD_DISASTER_THRESHOLD),
    }


def main():
    available = {}
    missing = []
    for k, p in PATHS.items():
        if not os.path.exists(p):
            missing.append(k)
            continue
        try:
            available[k] = json.load(open(p))
        except Exception as e:
            print(f"ERROR cargando {p}: {e}")
            missing.append(k)
    if not available:
        print("ERROR: no hay JSONs disponibles. Ejecuta tools/run_performance_grid.py --variant ...")
        sys.exit(1)
    if missing:
        print(f"⚠ JSONs faltantes (se omiten): {missing}")

    runs = {k: v['runs'] for k, v in available.items()}
    s    = {k: stats(runs[k]) for k in available}

    # ── KPIs (cada variante una tarjeta) ──────────────────────────────────
    kpi_cards = []
    for k in available.keys():
        st = s[k]
        if st is None:
            continue
        disaster_color = '#dc2626' if st['n_disaster'] > 0 else '#16a34a'
        kpi_cards.append(f"""
            <div class="kpi-card" style="border-left:4px solid {COLORS.get(k,'#000')}">
              <div class="kpi-name">{escape(k)}</div>
              <div class="kpi-stat"><span class="lbl">avg return</span>
                <span class="val" style="color:{color_by_sign(st['avg_ret'])}">{st['avg_ret']:+.2f}%</span></div>
              <div class="kpi-stat"><span class="lbl">avg Sharpe</span>
                <span class="val">{st['avg_sh']:+.2f}</span></div>
              <div class="kpi-stat"><span class="lbl">worst DD</span>
                <span class="val" style="color:{disaster_color}">{st['worst_dd']:+.2f}%</span></div>
              <div class="kpi-stat"><span class="lbl">% positivos</span>
                <span class="val">{st['n_pos']}/{st['n']}</span></div>
              <div class="kpi-stat"><span class="lbl">runs DD &lt; −40%</span>
                <span class="val" style="color:{disaster_color}">{st['n_disaster']}/{st['n']}</span></div>
              <div class="kpi-stat"><span class="lbl">avg trades/run</span>
                <span class="val">{st['avg_trades']:.1f}</span></div>
            </div>
        """)

    # ── Tabla por plazo ───────────────────────────────────────────────────
    # Plazos union de todas las variantes
    all_horizons = []
    for k in available:
        for r in runs[k]:
            h = r.get('horizon')
            if h and h not in all_horizons:
                all_horizons.append(h)
    # Orden por días
    all_h_days = {}
    for k in available:
        for r in runs[k]:
            if r.get('horizon') and r.get('horizon_days'):
                all_h_days[r['horizon']] = r['horizon_days']
    all_horizons = sorted(all_horizons, key=lambda h: all_h_days.get(h, 9999))

    table_rows = []
    for h in all_horizons:
        cells = [f"<td><b>{escape(h)}</b></td>"]
        for k in available:
            rs = [r for r in runs[k] if r.get('horizon') == h]
            if not rs:
                cells.append("<td>—</td>")
                continue
            avg_ret = sum(r['return_pct'] for r in rs) / len(rs)
            avg_dd  = sum(r['maxdd_pct'] for r in rs) / len(rs)
            avg_sh  = sum(r['sharpe']     for r in rs) / len(rs)
            worst_dd = min(r['maxdd_pct'] for r in rs)
            disaster_color = '#dc2626' if worst_dd < DD_DISASTER_THRESHOLD else '#64748b'
            cells.append(
                f"<td>"
                f"<div style='color:{color_by_sign(avg_ret)};font-weight:600'>{avg_ret:+.2f}%</div>"
                f"<div class='small'>Sh {avg_sh:+.2f}  ·  DD avg {avg_dd:+.1f}%</div>"
                f"<div class='small' style='color:{disaster_color}'>worst {worst_dd:+.1f}%</div>"
                f"</td>"
            )
        table_rows.append(f"<tr>{''.join(cells)}</tr>")

    # ── Bar chart agrupado por plazo (avg_return) ─────────────────────────
    bar_fig = go.Figure()
    for k in available:
        ys = []
        for h in all_horizons:
            rs = [r for r in runs[k] if r.get('horizon') == h]
            ys.append(sum(r['return_pct'] for r in rs) / len(rs) if rs else None)
        bar_fig.add_trace(go.Bar(
            x=all_horizons, y=ys, name=k, marker_color=COLORS.get(k, '#000'),
            text=[f"{y:+.1f}%" if y is not None else '' for y in ys],
            textposition='outside',
        ))
    bar_fig.update_layout(
        barmode='group', height=420, template='plotly_white',
        title='avg_return por plazo (todas las fechas agregadas)',
        margin=dict(l=40, r=20, t=50, b=40),
    )
    bar_html = pio.to_html(bar_fig, include_plotlyjs='cdn',
                            full_html=False, div_id='bar4')

    # ── Render HTML ───────────────────────────────────────────────────────
    style = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
             color:#1e293b; margin:0; padding:32px; background:#f8fafc; }
      h1 { color:#0f172a; }
      h2 { color:#334155; border-bottom:2px solid #e2e8f0; padding-bottom:6px; }
      .meta { color:#64748b; font-size:14px; margin-bottom:24px; }
      .kpi-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(230px,1fr));
                  gap:14px; margin: 22px 0; }
      .kpi-card { background:white; padding:16px; border-radius:10px;
                  box-shadow:0 1px 3px rgba(0,0,0,0.06); }
      .kpi-name { font-weight:700; font-size:14px; margin-bottom:8px; color:#0f172a;
                  letter-spacing:0.05em; }
      .kpi-stat { display:flex; justify-content:space-between; padding:3px 0;
                  font-size:13px; }
      .kpi-stat .lbl { color:#64748b; }
      .kpi-stat .val { font-weight:600; font-variant-numeric:tabular-nums; }
      table { border-collapse:collapse; width:100%; background:white;
              box-shadow:0 1px 3px rgba(0,0,0,0.06); margin:16px 0; }
      th { background:#1e293b; color:white; padding:10px 12px; text-align:left;
           font-size:13px; }
      td { padding:8px 12px; border-bottom:1px solid #e2e8f0;
           font-variant-numeric:tabular-nums; font-size:13px; }
      .small { font-size:11px; color:#64748b; }
    </style>
    """

    # Banner para bankruptcy
    n_disasters = {k: s[k]['n_disaster'] for k in available if s[k]}
    bankr_msg = ", ".join(f"{k}: {n}" for k, n in n_disasters.items())
    bankr_status = (
        '<div class="kpi-card" style="border-left:4px solid #16a34a">'
        f'<div class="kpi-name">✓ Bankruptcy guardrail</div>'
        f'<div class="kpi-stat">Ningún variant tiene runs con DD &lt; −40%.</div></div>'
        if all(n == 0 for n in n_disasters.values()) else
        '<div class="kpi-card" style="border-left:4px solid #dc2626">'
        f'<div class="kpi-name">⚠ Bankruptcy guardrail</div>'
        f'<div class="kpi-stat">Variantes con DD &lt; −40%: {escape(bankr_msg)}</div></div>'
    )

    th_cells = ''.join(f"<th>{escape(k)}</th>" for k in available)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Tier 5 — Dashboard 4-way</title>
  {style}
</head>
<body>
  <h1>Tier 5 — Dashboard comparativo 4-way</h1>
  <div class="meta">
    Generado: {datetime.now().isoformat(timespec='seconds')}<br>
    Variantes disponibles: {", ".join(available.keys())}<br>
    Faltantes: {", ".join(missing) if missing else "ninguna"}
  </div>

  <h2>KPIs por variante</h2>
  <div class="kpi-grid">
    {''.join(kpi_cards)}
    {bankr_status}
  </div>

  <h2>Resumen por plazo (avg_return / Sharpe / DD)</h2>
  <table>
    <thead><tr><th>plazo</th>{th_cells}</tr></thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>

  <h2>avg_return por plazo (bar chart)</h2>
  {bar_html}
</body>
</html>
"""

    with open(OUT_PATH, 'w') as f:
        f.write(html)
    print(f"[dashboard_compare4] HTML escrito: {OUT_PATH}")
    print(f"[dashboard_compare4] Bankruptcy: {bankr_msg}")


if __name__ == '__main__':
    main()
