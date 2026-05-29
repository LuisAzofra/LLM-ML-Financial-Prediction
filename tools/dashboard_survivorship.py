"""
Tier 5 — Dashboard del test de sesgo de supervivencia.

Lee `/tmp/survivorship_bias_result.json` (output de
`tools/measure_survivorship_bias.py`) y genera un HTML con:

  1. Resumen del experimento (fechas, plazos, seeds, AGGR_PLUS config).
  2. Tabla por plazo: famous vs broad avg_return, Δ_paired CI95.
  3. Heatmap (fecha × plazo): diferencia broad − famous (avg seeds).
  4. Tabla detallada de runs (todos los famous + broad por seed).
  5. Veredicto del gate de Fase 1.

Uso:
    .venv/bin/python -u tools/measure_survivorship_bias.py
    .venv/bin/python -u tools/dashboard_survivorship.py
    open /tmp/dashboard_survivorship.html
"""
import json
import os
import sys
from datetime import datetime
from html import escape

import plotly.graph_objects as go
import plotly.io as pio

IN_PATH  = os.environ.get('SURV_JSON', '/tmp/survivorship_bias_result.json')
OUT_PATH = os.environ.get('SURV_HTML', '/tmp/dashboard_survivorship.html')

GATE_RET_LO = -10.0  # pp — gate Fase 1 (PROGRESS Tier 5)


def color_by_sign(x: float, pos='#16a34a', neg='#dc2626', neu='#64748b') -> str:
    return pos if x > 0 else (neg if x < 0 else neu)


def main():
    if not os.path.exists(IN_PATH):
        print(f"ERROR: no existe {IN_PATH}. Corre tools/measure_survivorship_bias.py primero.")
        sys.exit(1)
    data = json.load(open(IN_PATH))
    cfg     = data.get('config', {})
    famous  = [r for r in data.get('famous', []) if 'error' not in r]
    broad   = [r for r in data.get('broad',  []) if 'error' not in r]
    summary = data.get('summary', {})

    # ── KPIs globales ─────────────────────────────────────────────────────
    if not famous or not broad or not summary:
        print(f"ERROR: JSON incompleto (famous={len(famous)}, broad={len(broad)}, "
              f"summary={len(summary)}).")
        sys.exit(1)

    horizons = list(summary.keys())

    avg_famous_ret = sum(r['return_pct'] for r in famous) / len(famous)
    avg_broad_ret  = sum(r['return_pct'] for r in broad)  / len(broad)
    avg_famous_dd  = sum(r['maxdd_pct']  for r in famous) / len(famous)
    avg_broad_dd   = sum(r['maxdd_pct']  for r in broad)  / len(broad)

    # ── Veredicto del gate ────────────────────────────────────────────────
    failed = []
    for h in horizons:
        d = summary[h]['delta_return_paired']
        if d['lo'] <= GATE_RET_LO:
            failed.append(h)
    gate_passed = not failed

    # ── Tabla por plazo ───────────────────────────────────────────────────
    rows_horizon = []
    for h in horizons:
        s = summary[h]
        f_ret = s['famous_avg_return']
        b_ret = s['broad_avg_return']
        d_ret = s['delta_return_paired']
        d_alp = s['delta_alpha_paired']
        d_dd  = s['delta_maxdd_paired']
        gate_ok = d_ret['lo'] > GATE_RET_LO
        rows_horizon.append(f"""
            <tr>
              <td><b>{escape(h)}</b></td>
              <td>{f_ret['mean']:+.2f}% [{f_ret['lo']:+.1f}, {f_ret['hi']:+.1f}]</td>
              <td>{b_ret['mean']:+.2f}% [{b_ret['lo']:+.1f}, {b_ret['hi']:+.1f}]</td>
              <td style="color:{color_by_sign(d_ret['point'])}">
                  {d_ret['point']:+.2f}pp [{d_ret['lo']:+.1f}, {d_ret['hi']:+.1f}]</td>
              <td style="color:{color_by_sign(d_alp['point'])}">
                  {d_alp['point']:+.2f}pp [{d_alp['lo']:+.1f}, {d_alp['hi']:+.1f}]</td>
              <td style="color:{color_by_sign(-d_dd['point'])}">
                  {d_dd['point']:+.2f}pp [{d_dd['lo']:+.1f}, {d_dd['hi']:+.1f}]</td>
              <td style="color:{'#16a34a' if gate_ok else '#dc2626'}; font-weight:600">
                  {'✓ PASA' if gate_ok else '✗ FALLA'}</td>
              <td>{s['famous_worst_dd']:+.2f}% / {s['broad_worst_dd']:+.2f}%</td>
            </tr>
        """)

    # ── Heatmap (fecha × plazo) ───────────────────────────────────────────
    start_dates = sorted({r['start_date'] for r in famous})
    z = []
    text = []
    for sd in start_dates:
        row_z, row_t = [], []
        for h in horizons:
            f_at = next((r for r in famous if r['start_date']==sd and r['horizon']==h), None)
            b_at = [r for r in broad  if r['start_date']==sd and r['horizon']==h]
            if not f_at or not b_at:
                row_z.append(None); row_t.append('')
                continue
            broad_avg = sum(r['return_pct'] for r in b_at) / len(b_at)
            d = broad_avg - f_at['return_pct']
            row_z.append(d)
            row_t.append(f"famous {f_at['return_pct']:+.1f}% → broad {broad_avg:+.1f}%<br>Δ {d:+.1f}pp")
        z.append(row_z); text.append(row_t)

    heatmap_fig = go.Figure(data=go.Heatmap(
        z=z, x=horizons, y=start_dates, text=text, hovertemplate='%{text}<extra></extra>',
        colorscale='RdYlGn', zmid=0, colorbar=dict(title='Δ pp')
    ))
    heatmap_fig.update_layout(
        title='Δ_return broad − famous (positivo = broad mejor)',
        height=420, template='plotly_white', margin=dict(l=80, r=20, t=50, b=40)
    )
    heatmap_html = pio.to_html(heatmap_fig, include_plotlyjs='cdn',
                                full_html=False, div_id='heatmap-surv')

    # ── Tabla detallada de runs (compacta) ────────────────────────────────
    detail_rows = []
    for sd in start_dates:
        for h in horizons:
            f_at = next((r for r in famous if r['start_date']==sd and r['horizon']==h), None)
            b_at = [r for r in broad if r['start_date']==sd and r['horizon']==h]
            if not f_at:
                continue
            broad_per_seed = ', '.join(
                f"s{r.get('seed','?')}: {r['return_pct']:+.1f}%" for r in b_at
            )
            broad_avg = sum(r['return_pct'] for r in b_at) / max(len(b_at), 1)
            d = broad_avg - f_at['return_pct']
            detail_rows.append(f"""
                <tr>
                  <td>{escape(sd)}</td>
                  <td>{escape(h)}</td>
                  <td>{f_at['return_pct']:+.2f}%</td>
                  <td>{f_at['bh_return_pct']:+.2f}%</td>
                  <td>{f_at['maxdd_pct']:+.2f}%</td>
                  <td>{f_at['total_trades']}</td>
                  <td>{escape(broad_per_seed)}</td>
                  <td style="color:{color_by_sign(d)}">{d:+.2f}pp</td>
                </tr>
            """)

    # ── Render HTML ───────────────────────────────────────────────────────
    style = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
             color:#1e293b; margin: 0; padding: 32px; background:#f8fafc; }
      h1 { color:#0f172a; }
      h2 { color:#334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }
      .meta { color:#64748b; font-size: 14px; margin-bottom: 24px; }
      .grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:16px; margin: 24px 0; }
      .kpi { background:#ffffff; padding:16px; border-radius:10px;
             box-shadow:0 1px 3px rgba(0,0,0,0.06); border:1px solid #e2e8f0;}
      .kpi .label { color:#64748b; font-size:12px; text-transform:uppercase;
                    letter-spacing:0.05em; }
      .kpi .value { font-size: 24px; font-weight: 700; margin-top: 4px;
                    font-variant-numeric: tabular-nums; }
      .verdict { padding: 16px 20px; border-radius: 10px; margin: 16px 0;
                 font-size: 16px; font-weight: 600; }
      .verdict.pass { background:#dcfce7; color:#166534; border:1px solid #86efac; }
      .verdict.fail { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; }
      table { border-collapse: collapse; width: 100%; background:white;
              box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin: 16px 0; }
      th { background: #1e293b; color:white; padding: 10px 12px; text-align: left;
           font-size: 13px; }
      td { padding: 8px 12px; border-bottom: 1px solid #e2e8f0;
           font-variant-numeric: tabular-nums; font-size: 13px; }
      tr:hover { background:#f1f5f9; }
      .small { font-size: 11px; color:#64748b; }
      pre { background:#0f172a; color:#cbd5e1; padding:14px; border-radius:8px;
            font-size:12px; overflow-x:auto; }
    </style>
    """

    verdict_html = (
        f'<div class="verdict pass">✓ Gate Fase 1 PASA en TODOS los plazos. '
        f'El bot tiene edge real — la rentabilidad NO se explica solo por el sesgo '
        f'de supervivencia (las 6 famosas).</div>'
        if gate_passed else
        f'<div class="verdict fail">✗ Gate Fase 1 FALLA en {", ".join(failed)}. '
        f'El sesgo de supervivencia está confirmado en estos plazos: el bot rinde '
        f'significativamente peor en universo amplio. Ver Fase 4 (recalibración).</div>'
    )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Tier 5 — Test sesgo de supervivencia</title>
  {style}
</head>
<body>
  <h1>Tier 5 — Test cuantitativo de sesgo de supervivencia</h1>
  <div class="meta">
    Generado: {datetime.now().isoformat(timespec='seconds')}<br>
    Backtests famous: {len(famous)} · broad: {len(broad)} ({len(cfg.get('seeds', []))} seeds)<br>
    Plazos: {", ".join(horizons)} · Fechas inicio: {len(cfg.get('start_dates', []))}<br>
    Variante AGGR_PLUS · n_stocks={cfg.get('n_stocks','?')} · n_crypto={cfg.get('n_crypto','?')}<br>
    Tiempo total: {cfg.get('elapsed_sec', '?')} s
  </div>

  {verdict_html}

  <h2>KPIs globales (todos los plazos agregados)</h2>
  <div class="grid">
    <div class="kpi">
      <div class="label">avg return famous</div>
      <div class="value" style="color:{color_by_sign(avg_famous_ret)}">{avg_famous_ret:+.2f}%</div>
    </div>
    <div class="kpi">
      <div class="label">avg return broad</div>
      <div class="value" style="color:{color_by_sign(avg_broad_ret)}">{avg_broad_ret:+.2f}%</div>
    </div>
    <div class="kpi">
      <div class="label">Δ avg return</div>
      <div class="value" style="color:{color_by_sign(avg_broad_ret-avg_famous_ret)}">{avg_broad_ret - avg_famous_ret:+.2f}pp</div>
    </div>
    <div class="kpi">
      <div class="label">avg MaxDD broad − famous</div>
      <div class="value" style="color:{color_by_sign(-(avg_broad_dd-avg_famous_dd))}">{avg_broad_dd - avg_famous_dd:+.2f}pp</div>
    </div>
  </div>

  <h2>Por plazo — test paired bootstrap</h2>
  <table>
    <thead>
      <tr>
        <th>Plazo</th>
        <th>famous avg_ret CI95</th>
        <th>broad avg_ret CI95</th>
        <th>Δ_return paired CI95</th>
        <th>Δ_alpha vs B&H</th>
        <th>Δ_MaxDD</th>
        <th>Gate (lo &gt; -10pp)</th>
        <th>worst_DD famous / broad</th>
      </tr>
    </thead>
    <tbody>{''.join(rows_horizon)}</tbody>
  </table>

  <h2>Heatmap broad − famous (fecha × plazo)</h2>
  {heatmap_html}

  <h2>Detalle de runs (cada celda)</h2>
  <table>
    <thead>
      <tr>
        <th>start_date</th><th>plazo</th>
        <th>famous_ret</th><th>famous_BH</th><th>famous_DD</th><th>famous_trades</th>
        <th>broad por seed</th><th>Δ_paired</th>
      </tr>
    </thead>
    <tbody>{''.join(detail_rows)}</tbody>
  </table>

  <h2>Configuración exacta (reproducible)</h2>
  <pre>{escape(json.dumps(cfg, indent=2)[:3000])}</pre>
</body>
</html>
"""

    with open(OUT_PATH, 'w') as f:
        f.write(html)
    print(f"[dashboard_survivorship] HTML escrito: {OUT_PATH}")
    print(f"[dashboard_survivorship] Veredicto: {'PASA' if gate_passed else 'FALLA en ' + ','.join(failed)}")


if __name__ == '__main__':
    main()
