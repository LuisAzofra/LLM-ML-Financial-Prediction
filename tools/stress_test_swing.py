"""
Tier 5 Fase 3 — Stress test de swing mode con guardrails de quiebra.

Corre SWING_DYNAMIC y SWING_BROAD sobre 4 ventanas de stress documentadas:
  · 2020-02-15 → 2020-04-15  (COVID crash, S&P -34% en 5 semanas)
  · 2018-09-15 → 2018-12-31  (Q4 2018 -20%, gap risk overnight)
  · 2022-01-01 → 2022-06-30  (bear primario 2022, NDX -28%)
  · 2021-05-01 → 2021-07-31  (crypto crash mayo 2021, BTC -50%)

Verifica los criterios anti-quiebra:
  · `worst_dd > −40%` → si NO se cumple, fail
  · `equity_min < 0.40 * initial_capital` → fail
  · ningún error en backtest (status=success)

Uso:
    PORT=5057 .venv/bin/python -u api.py &
    .venv/bin/python -u tools/stress_test_swing.py
"""
import json
import os
import sys
import time
from urllib import request as urlreq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

API = os.environ.get('API_URL', 'http://localhost:5057')

STRESS_WINDOWS = [
    ('COVID_CRASH_2020',     '2020-02-15', '2020-04-15'),
    ('Q4_2018_GAP_RISK',     '2018-09-15', '2018-12-31'),
    ('BEAR_2022_H1',         '2022-01-01', '2022-06-30'),
    ('CRYPTO_CRASH_MAY2021', '2021-05-01', '2021-07-31'),
]

INIT_CAP = 100_000.0

SWING_BASE = {
    'mode': 'swing',
    'initial_capital': INIT_CAP,
    'allow_short': False,
    'signal_percentile': 0.70,
    'kelly_scale': 0.30,
    'trend_reverse_exit': True,
    'use_llm': False,
    'topn_rotation': True,
    'topn_value': 5,
    'max_concurrent': 5,
    'include_etfs': True,
    'max_holding_days_swing': 14,
    'min_holding_days_swing': 1,
    'conf_floor': 0.50,
}

VARIANTS = {
    'SWING_FAMOUS':  SWING_BASE,
    'SWING_BROAD_42': {**SWING_BASE,
                       'universe_mode': 'broad_random',
                       'universe_seed': 42,
                       'universe_size_stocks': 30,
                       'universe_size_crypto': 20},
}

DD_DISASTER  = -40.0
EQUITY_FLOOR = 0.40 * INIT_CAP


def call(start, end, body_extra):
    body = {**body_extra, 'start_date': start, 'end_date': end}
    req = urlreq.Request(
        f"{API}/api/paper/autonomous-backtest",
        method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urlreq.urlopen(req, timeout=600) as r:
        payload = json.loads(r.read())
    elapsed = time.time() - t0
    if payload.get('status') != 'success':
        raise RuntimeError(payload.get('error', 'no error msg'))
    p = payload['performance']
    eq = payload.get('equity_curve', {}).get('values', [])
    return {
        'return':   p['total_return_pct'],
        'maxdd':    p['max_drawdown_pct'],
        'sharpe':   p['sharpe_ratio'],
        'trades':   p['total_trades'],
        'equity_min': float(min(eq)) if eq else float(p['initial_capital']),
        'equity_final': float(eq[-1]) if eq else float(p['initial_capital']),
        'elapsed':  elapsed,
    }


def main():
    print(f"Stress test — {len(STRESS_WINDOWS)} ventanas × {len(VARIANTS)} variantes "
          f"= {len(STRESS_WINDOWS)*len(VARIANTS)} backtests")
    print(f"Guardrails: worst_DD > {DD_DISASTER}%  ·  equity_min > ${EQUITY_FLOOR:,.0f}")
    print()

    results = {v: [] for v in VARIANTS}
    fails = []
    for label, sd, ed in STRESS_WINDOWS:
        print(f"── {label}  {sd} → {ed} ──")
        for vname, vbody in VARIANTS.items():
            try:
                r = call(sd, ed, vbody)
                r['label'] = label; r['start'] = sd; r['end'] = ed
                # Aplicar guardrails
                dd_ok  = r['maxdd'] > DD_DISASTER
                eq_ok  = r['equity_min'] > EQUITY_FLOOR
                r['guardrails'] = {'dd_ok': dd_ok, 'eq_ok': eq_ok,
                                    'fail': not (dd_ok and eq_ok)}
                results[vname].append(r)
                tag = '✓' if not r['guardrails']['fail'] else '✗ FAIL'
                print(f"  {vname:<18}  ret={r['return']:+7.2f}%  DD={r['maxdd']:+6.2f}%  "
                      f"eq_min=${r['equity_min']:>7,.0f}  trades={r['trades']:>3}  {tag}  "
                      f"({r['elapsed']:.0f}s)")
                if r['guardrails']['fail']:
                    fails.append((vname, label, r))
            except Exception as ex:
                print(f"  {vname:<18}  CRASH · {ex}")
                fails.append((vname, label, {'error': str(ex)}))
        sys.stdout.flush()
        print()

    # Resumen
    print('═' * 80)
    print('RESUMEN')
    print('═' * 80)
    for vname in VARIANTS:
        rs = results[vname]
        if not rs:
            print(f"  {vname:<18}  sin datos")
            continue
        worst_dd = min(r['maxdd'] for r in rs)
        worst_eq = min(r['equity_min'] for r in rs)
        n_fail   = sum(1 for r in rs if r['guardrails']['fail'])
        status = '✓ PASA TODOS' if n_fail == 0 else f'✗ FALLA en {n_fail}/{len(rs)}'
        print(f"  {vname:<18}  worst_DD={worst_dd:+.2f}%  "
              f"worst_eq=${worst_eq:>7,.0f}  {status}")

    out_path = '/tmp/stress_test_swing_result.json'
    with open(out_path, 'w') as f:
        json.dump({
            'guardrails': {'dd_disaster': DD_DISASTER, 'equity_floor': EQUITY_FLOOR},
            'variants':    {k: v for k, v in VARIANTS.items()},
            'windows':     STRESS_WINDOWS,
            'results':     results,
            'fails':       [(v, l, r) for v, l, r in fails],
        }, f, indent=2, default=str)
    print(f"\n→ JSON: {out_path}")

    # Exit code para CI / scripting downstream
    if fails:
        print(f"\n⚠ Stress test detectó {len(fails)} fallos del guardrail. NO mergear "
              f"swing como variante recomendada hasta resolverlos (Phase 4).")
        sys.exit(1)


if __name__ == '__main__':
    main()
