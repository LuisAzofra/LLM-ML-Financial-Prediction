"""
Tier 5 Phase 4 — Sweep de signal_percentile × kelly_scale específico para
universo broad en mode='swing'. Hipótesis: el 0.70/0.30 está calibrado para
distribución de scores de tech famosas; en broad puede que se necesite más
conservador para reducir DD sin sacrificar return.

Métrica de selección: Calmar (avg_ret / |worst_DD|) — penaliza DD elevado
sin descartar retornos altos.

Output: /tmp/sweep_swing_broad_result.json + tabla en stdout.

Uso:
    PORT=5057 .venv/bin/python -u api.py &
    .venv/bin/python -u tools/sweep_swing_broad.py
"""
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from urllib import request as urlreq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

API = os.environ.get('API_URL', 'http://localhost:5057')
N_WINDOWS = int(os.environ.get('N_WINDOWS', '8'))
WIN_YEARS = float(os.environ.get('WIN_YEARS', '0.5'))
SEED = 42

EARLIEST = datetime(2018, 1, 1)
LATEST   = datetime(2024, 4, 1)

CONFIGS = [
    # (label, signal_percentile, kelly_scale)
    ('s70_k20', 0.70, 0.20),
    ('s70_k25', 0.70, 0.25),
    ('s70_k30', 0.70, 0.30),  # default actual de SWING_BROAD_42
    ('s80_k20', 0.80, 0.20),
    ('s80_k25', 0.80, 0.25),
    ('s80_k30', 0.80, 0.30),
]

BASE = {
    'mode': 'swing',
    'initial_capital': 100_000,
    'allow_short': False,
    'trend_reverse_exit': True,
    'use_llm': False,
    'topn_rotation': True,
    'topn_value': 5,
    'max_concurrent': 5,
    'include_etfs': True,
    'max_holding_days_swing': 14,
    'min_holding_days_swing': 1,
    'conf_floor': 0.50,
    'universe_mode': 'broad_random',
    'universe_seed': 42,
    'universe_size_stocks': 30,
    'universe_size_crypto': 20,
}


def random_window(rng):
    days = rng.randint(0, (LATEST - EARLIEST).days - int(365 * WIN_YEARS))
    s = EARLIEST + timedelta(days=days)
    e = s + timedelta(days=int(365 * WIN_YEARS))
    return s.strftime('%Y-%m-%d'), e.strftime('%Y-%m-%d')


def call(start, end, body_extra):
    body = {**BASE, **body_extra, 'start_date': start, 'end_date': end}
    req = urlreq.Request(
        f"{API}/api/paper/autonomous-backtest", method='POST',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
    )
    t0 = time.time()
    with urlreq.urlopen(req, timeout=600) as r:
        payload = json.loads(r.read())
    if payload.get('status') != 'success':
        raise RuntimeError(payload.get('error', '?'))
    p = payload['performance']
    return {
        'return': p['total_return_pct'],
        'maxdd':  p['max_drawdown_pct'],
        'sharpe': p['sharpe_ratio'],
        'trades': p['total_trades'],
        'elapsed': time.time() - t0,
    }


def main():
    rng = random.Random(SEED)
    windows = [random_window(rng) for _ in range(N_WINDOWS)]
    print(f"Sweep SWING_BROAD: {len(CONFIGS)} configs × {N_WINDOWS} ventanas {WIN_YEARS}y "
          f"= {len(CONFIGS)*N_WINDOWS} backtests")
    print()

    results = {label: [] for label, _, _ in CONFIGS}
    for i, (s, e) in enumerate(windows, 1):
        print(f"── W{i}/{N_WINDOWS}  {s} → {e} ──")
        for label, sp, ks in CONFIGS:
            try:
                r = call(s, e, {'signal_percentile': sp, 'kelly_scale': ks})
                results[label].append(r)
                print(f"  {label:<10}  ret={r['return']:+7.2f}%  DD={r['maxdd']:+6.2f}%  "
                      f"Sh={r['sharpe']:+.2f}  ({r['elapsed']:.1f}s)")
            except Exception as ex:
                print(f"  {label:<10}  FAIL · {ex}")
        sys.stdout.flush()
        print()

    # Agregados
    print('═' * 90)
    print(f"{'config':<10} {'avg_ret':>9} {'worst_DD':>9} {'avg_Sharpe':>11} "
          f"{'Calmar':>7} {'%pos':>6} {'avg_trades':>11}")
    print('─' * 90)
    summary = []
    for label, sp, ks in CONFIGS:
        rs = results[label]
        if not rs:
            continue
        avg_ret  = sum(r['return'] for r in rs) / len(rs)
        worst_dd = min(r['maxdd'] for r in rs)
        avg_sh   = sum(r['sharpe'] for r in rs) / len(rs)
        n_pos    = sum(1 for r in rs if r['return'] > 0)
        avg_tr   = sum(r['trades'] for r in rs) / len(rs)
        calmar   = avg_ret / abs(worst_dd) if abs(worst_dd) > 1e-6 else 0.0
        summary.append({
            'label': label, 'signal_pct': sp, 'kelly_scale': ks,
            'avg_ret': avg_ret, 'worst_dd': worst_dd,
            'avg_sharpe': avg_sh, 'calmar': calmar,
            'pct_pos': n_pos / len(rs) * 100, 'avg_trades': avg_tr,
            'n': len(rs),
        })
        print(f"{label:<10} {avg_ret:+8.2f}% {worst_dd:+7.2f}%  {avg_sh:+10.2f}  "
              f"{calmar:+6.3f}  {n_pos/len(rs)*100:>5.1f}%  {avg_tr:>10.1f}")

    # Pickeamos el mejor por Calmar (return/|DD|)
    best = max(summary, key=lambda r: r['calmar'])
    print()
    print(f"GANADOR (por Calmar): {best['label']}  "
          f"(signal_pct={best['signal_pct']}, kelly_scale={best['kelly_scale']})")
    print(f"  avg_ret={best['avg_ret']:+.2f}%  worst_DD={best['worst_dd']:+.2f}%  "
          f"Calmar={best['calmar']:+.3f}  %pos={best['pct_pos']:.1f}%")

    # Comparar con default (s70_k30)
    default = next(r for r in summary if r['label'] == 's70_k30')
    print(f"\nvs default (s70_k30):")
    print(f"  Δ_avg_ret  = {best['avg_ret']-default['avg_ret']:+.2f}pp")
    print(f"  Δ_worst_DD = {best['worst_dd']-default['worst_dd']:+.2f}pp")
    print(f"  Δ_Calmar   = {best['calmar']-default['calmar']:+.3f}")

    out = {
        'seed': SEED,
        'n_windows': N_WINDOWS,
        'win_years': WIN_YEARS,
        'configs': CONFIGS,
        'summary': summary,
        'best': best,
    }
    with open('/tmp/sweep_swing_broad_result.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n→ /tmp/sweep_swing_broad_result.json")


if __name__ == '__main__':
    main()
