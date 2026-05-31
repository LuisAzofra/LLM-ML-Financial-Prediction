"""
ml_llm_news_backtest.py — estrategia HONESTA ML+LLM con noticias point-in-time.

Combina:
  · ML  : RandomForest que predice P(QQQ suba en 21 días) a partir de features
          de precio (momentum 1/3/6m, distancia a SMA200, volatilidad 20d).
  · LLM : sentimiento FinBERT de noticias NASDAQ REALES y POINT-IN-TIME
          (data/news_sentiment_nasdaq.csv, generado por build_news_sentiment.py)
          como feature adicional del modelo.

Decisión: long/flat sobre QQQ. Estar LONG cuando el modelo no es bajista; si no,
a efectivo. La estructura long/flat sobre un índice alcista es estructuralmente
positiva — el ML+LLM decide *cuándo* estar dentro/fuera (overlay de riesgo).

Honestidad / causalidad:
  · Split temporal: el modelo se ENTRENA solo con TRAIN (2019-2020) y se evalúa
    OUT-OF-SAMPLE en TEST (2021-2023). Cero solapamiento.
  · El sentimiento del día t usa solo noticias con fecha ≤ t (rolling causal).
  · El target mira 21 días adelante SOLO en train (nunca en la decisión OOS).
  · El efectivo NO renta (rf=0) → resultado conservador.

Uso:
    PYTHONPATH=. .venv/bin/python tools/ml_llm_news_backtest.py --mode riskoff --n-windows 8 --seed 42
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier

SENT_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "news_sentiment_nasdaq.csv")

TRAIN_START, TRAIN_END = "2019-01-01", "2020-12-31"
TEST_START,  TEST_END  = "2021-01-01", "2023-12-31"
HORIZONS = [("1M", 30), ("6M", 182), ("1Y", 365), ("2Y", 730)]
FEATURES = ["mom_21", "mom_63", "mom_126", "dist_sma200", "vol_20", "sent_20"]


def load_qqq():
    from data.data_loader import FinancialDataLoader
    loader = FinancialDataLoader()
    df = loader.download_data("QQQ", "2018-01-01", "2024-02-01", asset_type="stock")
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[["Close"]].copy()
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df


def build_frame():
    px = load_qqq()
    sent = pd.read_csv(SENT_CSV, parse_dates=["date"]).set_index("date").sort_index()
    # Sentimiento a días de trading: reindex + ffill (causal) + media móvil 20d.
    s = sent["sent_net"].reindex(px.index.union(sent.index)).sort_index().ffill()
    s = s.reindex(px.index)
    f = pd.DataFrame(index=px.index)
    c = px["Close"].astype(float)
    f["mom_21"]  = c / c.shift(21) - 1
    f["mom_63"]  = c / c.shift(63) - 1
    f["mom_126"] = c / c.shift(126) - 1
    f["dist_sma200"] = c / c.rolling(200, min_periods=100).mean() - 1
    f["vol_20"] = c.pct_change().rolling(20).std() * np.sqrt(252)
    # Sentimiento: media de los últimos 20 días, desplazado 1 día (estrictamente causal).
    f["sent_20"] = s.rolling(20, min_periods=3).mean().shift(1)
    f["close"] = c
    f["ret_1"] = c.pct_change()
    f["fwd21"] = c.shift(-21) / c - 1   # solo para entrenar
    return f


def train_model(f, seed):
    tr = f.loc[TRAIN_START:TRAIN_END].dropna(subset=FEATURES + ["fwd21"]).copy()
    y = (tr["fwd21"] > 0).astype(int)
    clf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=20,
                                 random_state=seed, n_jobs=-1)
    clf.fit(tr[FEATURES], y)
    return clf


def signal_proba(f, clf):
    valid = f.dropna(subset=FEATURES)
    p = pd.Series(np.nan, index=f.index)
    p.loc[valid.index] = clf.predict_proba(valid[FEATURES])[:, 1]
    return p


def run_window(f, p_up, sd, ed, mode):
    w = f.loc[sd:ed]
    if len(w) < 5:
        return None
    thr = 0.45 if mode == "riskoff" else 0.50
    in_mkt = (p_up.loc[w.index].fillna(0.5) >= thr) if mode != "riskoff" \
        else (p_up.loc[w.index].fillna(1.0) >= thr)  # riskoff: por defecto dentro
    pos = in_mkt.astype(float).shift(1).fillna(0.0)   # decidir en t, aplicar en t+1
    rets = w["ret_1"].fillna(0.0)
    strat = (pos * rets)                               # efectivo renta 0
    eq = float((1 + strat).prod())
    bh = float(w["close"].iloc[-1] / w["close"].iloc[0])
    dd = ((1 + strat).cumprod() / (1 + strat).cumprod().cummax() - 1).min()
    days_in = float(pos.mean())
    return {"ret": (eq - 1) * 100, "bh": (bh - 1) * 100, "maxdd": dd * 100,
            "pct_in_market": days_in * 100}


def sample_starts(h_days, n, seed, h_index):
    lo = pd.Timestamp(TEST_START)
    hi = pd.Timestamp(TEST_END) - pd.Timedelta(days=h_days)
    span = (hi - lo).days
    if span <= 0:
        return []
    rng = np.random.default_rng(seed + h_index)
    offs = sorted(set(int(x) for x in rng.integers(0, span + 1, size=n * 4)))[:n]
    return [lo + pd.Timedelta(days=int(o)) for o in offs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["balanced", "riskoff"], default="riskoff")
    ap.add_argument("--n-windows", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    f = build_frame()
    clf = train_model(f, args.seed)
    p_up = signal_proba(f, clf)

    imp = dict(zip(FEATURES, clf.feature_importances_.round(3)))
    print(f"Estrategia ML+LLM (mode={args.mode}) — TRAIN {TRAIN_START}..{TRAIN_END}  "
          f"TEST OOS {TEST_START}..{TEST_END}")
    print(f"Importancia features (incluye sent_20 = FinBERT): {imp}")
    print(f"  → peso del sentimiento FinBERT en el modelo: {imp['sent_20']:.3f}")
    print()
    print(f"{'plazo':<6}{'ML+LLM medio':>14}{'mediana':>10}{'%pos':>6}"
          f"{'QQQ b&h medio':>15}{'maxDD medio':>13}{'%t dentro':>11}")
    overall = []
    for hi, (hl, hd) in enumerate(HORIZONS):
        starts = sample_starts(hd, args.n_windows, args.seed, hi)
        rows = []
        for sd in starts:
            r = run_window(f, p_up, sd.strftime("%Y-%m-%d"),
                           (sd + pd.Timedelta(days=hd)).strftime("%Y-%m-%d"), args.mode)
            if r:
                rows.append(r)
        if not rows:
            print(f"{hl:<6}  (sin ventanas)")
            continue
        rets = [r["ret"] for r in rows]
        pos = 100 * sum(1 for x in rets if x > 0) / len(rets)
        print(f"{hl:<6}{np.mean(rets):>+12.2f}%{np.median(rets):>+9.2f}%{pos:>5.0f}%"
              f"{np.mean([r['bh'] for r in rows]):>+13.2f}%"
              f"{np.mean([r['maxdd'] for r in rows]):>+12.2f}%"
              f"{np.mean([r['pct_in_market'] for r in rows]):>10.0f}%")
        overall.extend(rets)
    print()
    print(f"GLOBAL OOS: media {np.mean(overall):+.2f}%  mediana {np.median(overall):+.2f}%  "
          f"%ventanas positivas {100*sum(1 for x in overall if x>0)/len(overall):.0f}%  (n={len(overall)})")


if __name__ == "__main__":
    main()
