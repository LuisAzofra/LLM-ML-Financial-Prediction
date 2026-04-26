"""
Compara modo `ml` (V5 — ensemble + R² gate) vs modo `trend`
(Faber/Antonacci puro) sobre N ventanas aleatorias de 2 años del pasado.

Para cada ventana se ejecutan los DOS modos y se comparan:
  · total_return_pct
  · max_drawdown_pct
  · sharpe_ratio / profit_factor / win_rate_pct
  · alpha vs B&H (mismo universo en ambos modos)

Reglas para que la comparación sea justa y leak-free:
  · Las ventanas se eligen entre 2018-01-01 y 2024-04-01, dejando ≥4 años
    de historia antes de cada ventana (que el ML necesita para entrenar).
  · Mismo `start_date`/`end_date` para los dos modos en cada iteración.
  · Mismo capital, sin shorts, mismos parámetros operativos.
  · Semilla fija para reproducibilidad.
"""
import json
import random
import statistics
import sys
import time
from datetime import datetime, timedelta
from urllib import request as urlreq

API = "http://localhost:5057"
N_WINDOWS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
WINDOW_YEARS = 2
SEED = 42
random.seed(SEED)

# Periodo desde el que sortear el comienzo de cada ventana.
# Final 2024-04-01 → la última ventana posible es 2024-04-01 → 2026-04-01.
EARLIEST = datetime(2018, 1, 1)
LATEST   = datetime(2024, 4, 1)

DELTA_DAYS = (LATEST - EARLIEST).days

def random_window():
    days = random.randint(0, DELTA_DAYS)
    start = EARLIEST + timedelta(days=days)
    end   = start + timedelta(days=365 * WINDOW_YEARS)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')

def call(mode, start, end):
    body = {
        "mode":              mode,
        "start_date":        start,
        "end_date":          end,
        "initial_capital":   100_000,
        "allow_short":       False,
        "signal_percentile": 0.82,
        "kelly_scale":       0.22,
    }
    req = urlreq.Request(
        f"{API}/api/paper/autonomous-backtest",
        method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urlreq.urlopen(req, timeout=1200) as r:
        payload = json.loads(r.read())
    elapsed = time.time() - t0
    if payload.get('status') != 'success':
        raise RuntimeError(f"{mode} {start}→{end} → {payload}")
    p = payload['performance']
    return {
        'return':  p['total_return_pct'],
        'bh':      p['buy_hold_return_pct'],
        'alpha':   p['total_return_pct'] - p['buy_hold_return_pct'],
        'maxdd':   p['max_drawdown_pct'],
        'sharpe':  p['sharpe_ratio'],
        'sortino': p['sortino_ratio'],
        'pf':      p['profit_factor'],
        'wr':      p['win_rate_pct'],
        'trades':  p['total_trades'],
        'elapsed': elapsed,
    }

def fmt_row(label, m):
    return (f"{label:<8}{m['return']:+8.2f}%{m['bh']:+8.2f}%{m['alpha']:+8.2f}pp"
            f"{m['maxdd']:+8.2f}%{m['sharpe']:7.2f}{m['pf']:7.2f}"
            f"{m['wr']:6.1f}%{m['trades']:6d}")

def main():
    windows = [random_window() for _ in range(N_WINDOWS)]
    print("Ventanas (seed=", SEED, "):", sep='')
    for i, (s, e) in enumerate(windows, 1):
        print(f"  W{i}  {s} → {e}")
    print()

    rows = []
    for i, (s, e) in enumerate(windows, 1):
        print(f"── Ventana {i}/{N_WINDOWS}  {s} → {e} ──")
        try:
            m = call('ml', s, e)
            rows.append(('ml', i, s, e, m))
            print(f"  ml    : ret={m['return']:+7.2f}%  alpha={m['alpha']:+6.2f}pp  DD={m['maxdd']:+6.2f}%  PF={m['pf']:5.2f}  trades={m['trades']:3d}  ({m['elapsed']:.0f}s)")
        except Exception as ex:
            print(f"  ml    : FAIL · {ex}")
        try:
            m = call('trend', s, e)
            rows.append(('trend', i, s, e, m))
            print(f"  trend : ret={m['return']:+7.2f}%  alpha={m['alpha']:+6.2f}pp  DD={m['maxdd']:+6.2f}%  PF={m['pf']:5.2f}  trades={m['trades']:3d}  ({m['elapsed']:.0f}s)")
        except Exception as ex:
            print(f"  trend : FAIL · {ex}")
        print()

    # ── tabla detallada ──
    print()
    print("═" * 98)
    print("DETALLE POR VENTANA")
    print("═" * 98)
    print(f"{'mode':<8}{'ret':>9}{'B&H':>9}{'alpha':>9}{'maxDD':>9}{'sharpe':>7}{'PF':>7}{'WR':>7}{'trades':>7}")
    print("─" * 98)
    for mode, i, s, e, m in rows:
        print(f"W{i} {mode:<5}{m['return']:+8.2f}%{m['bh']:+8.2f}%{m['alpha']:+8.2f}pp{m['maxdd']:+8.2f}%{m['sharpe']:7.2f}{m['pf']:7.2f}{m['wr']:6.1f}%{m['trades']:6d}")

    # ── agregados ──
    def agg(mode):
        ms = [r[4] for r in rows if r[0] == mode]
        if not ms: return None
        return {
            'n':           len(ms),
            'avg_return':  statistics.mean(m['return'] for m in ms),
            'med_return':  statistics.median(m['return'] for m in ms),
            'avg_bh':      statistics.mean(m['bh']     for m in ms),
            'avg_alpha':   statistics.mean(m['alpha']  for m in ms),
            'avg_maxdd':   statistics.mean(m['maxdd']  for m in ms),
            'avg_sharpe':  statistics.mean(m['sharpe'] for m in ms),
            'avg_pf':      statistics.mean(m['pf']     for m in ms if m['pf'] < 50),
            'avg_wr':      statistics.mean(m['wr']     for m in ms),
            'avg_trades':  statistics.mean(m['trades'] for m in ms),
            'pct_pos':     100 * sum(1 for m in ms if m['return']  > 0)        / len(ms),
            'pct_beat_bh': 100 * sum(1 for m in ms if m['alpha']   > 0)        / len(ms),
            'worst_dd':    min(m['maxdd']  for m in ms),
            'best_return': max(m['return'] for m in ms),
            'worst_return':min(m['return'] for m in ms),
        }

    print()
    print("═" * 98)
    print("AGREGADO (promedio entre ventanas)")
    print("═" * 98)
    a_ml = agg('ml')
    a_tr = agg('trend')
    if a_ml and a_tr:
        rows_agg = [
            ('# ventanas',       f"{a_ml['n']}",                      f"{a_tr['n']}"),
            ('avg return',       f"{a_ml['avg_return']:+.2f}%",       f"{a_tr['avg_return']:+.2f}%"),
            ('median return',    f"{a_ml['med_return']:+.2f}%",       f"{a_tr['med_return']:+.2f}%"),
            ('avg B&H',          f"{a_ml['avg_bh']:+.2f}%",           f"{a_tr['avg_bh']:+.2f}%"),
            ('avg alpha vs B&H', f"{a_ml['avg_alpha']:+.2f}pp",       f"{a_tr['avg_alpha']:+.2f}pp"),
            ('avg maxDD',        f"{a_ml['avg_maxdd']:+.2f}%",        f"{a_tr['avg_maxdd']:+.2f}%"),
            ('worst maxDD',      f"{a_ml['worst_dd']:+.2f}%",         f"{a_tr['worst_dd']:+.2f}%"),
            ('avg Sharpe',       f"{a_ml['avg_sharpe']:.2f}",         f"{a_tr['avg_sharpe']:.2f}"),
            ('avg PF',           f"{a_ml['avg_pf']:.2f}",             f"{a_tr['avg_pf']:.2f}"),
            ('avg WR',           f"{a_ml['avg_wr']:.1f}%",            f"{a_tr['avg_wr']:.1f}%"),
            ('avg trades',       f"{a_ml['avg_trades']:.0f}",         f"{a_tr['avg_trades']:.0f}"),
            ('% windows ret>0',  f"{a_ml['pct_pos']:.0f}%",           f"{a_tr['pct_pos']:.0f}%"),
            ('% beats B&H',      f"{a_ml['pct_beat_bh']:.0f}%",       f"{a_tr['pct_beat_bh']:.0f}%"),
            ('best window',      f"{a_ml['best_return']:+.2f}%",      f"{a_tr['best_return']:+.2f}%"),
            ('worst window',     f"{a_ml['worst_return']:+.2f}%",     f"{a_tr['worst_return']:+.2f}%"),
        ]
        print(f"{'metric':<22}{'ML (V5)':>22}{'TREND':>22}")
        print('─' * 98)
        for label, v_ml, v_tr in rows_agg:
            print(f"{label:<22}{v_ml:>22}{v_tr:>22}")

        # ── veredicto ──
        print()
        print('═' * 98)
        # Ranking por: promedio retorno + % windows positives + alpha
        score_ml = a_ml['avg_return'] * 0.5 + a_ml['avg_alpha'] * 0.3 + a_ml['pct_pos'] * 0.05
        score_tr = a_tr['avg_return'] * 0.5 + a_tr['avg_alpha'] * 0.3 + a_tr['pct_pos'] * 0.05
        print(f"Score compuesto (0.5·avg_ret + 0.3·avg_alpha + 0.05·%pos):")
        print(f"  ML    : {score_ml:+7.2f}")
        print(f"  TREND : {score_tr:+7.2f}")
        winner = 'TREND' if score_tr > score_ml else 'ML'
        print()
        print(f"GANADOR → {winner}")
        print('═' * 98)

    # Guardar JSON detallado para auditoría
    out = {
        'seed': SEED,
        'n_windows': N_WINDOWS,
        'window_years': WINDOW_YEARS,
        'rows': [
            {'mode': mode, 'window': i, 'start': s, 'end': e, **m}
            for mode, i, s, e, m in rows
        ],
        'aggregates': {'ml': a_ml, 'trend': a_tr},
    }
    with open('/tmp/compare_modes_result.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nJSON detallado → /tmp/compare_modes_result.json")

if __name__ == '__main__':
    main()
