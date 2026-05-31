"""
build_news_sentiment.py — construye una serie de sentimiento de mercado DIARIO
y POINT-IN-TIME puntuando con FinBERT noticias NASDAQ históricas reales.

Fuente: dataset `benstaf/nasdaq_news` (HuggingFace) — 61.827 titulares NASDAQ
con fecha, 2019-2023. Cada titular se puntúa con FinBERT (net = P(pos)−P(neg))
y se agrega por fecha. El resultado es una serie causal: el sentimiento del día
D usa solo titulares publicados el día D.

Salida: data/news_sentiment_nasdaq.csv con columnas
    date, sent_net, sent_pos, sent_neg, n_titles

Uso:
    PYTHONPATH=. .venv/bin/python tools/build_news_sentiment.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PARQUET = os.environ.get(
    "NASDAQ_NEWS_PARQUET",
    "/tmp/TFG/newsdata/data/train-00000-of-00001.parquet",
)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "news_sentiment_nasdaq.csv")


def main():
    df = pd.read_parquet(PARQUET)
    df["_d"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["_d", "Article_title"])
    df["Article_title"] = df["Article_title"].astype(str).str.strip()
    df = df[df["Article_title"].str.len() > 0]
    print(f"titulares con fecha: {len(df)}  ({df['_d'].min().date()} → {df['_d'].max().date()})")

    # Dedup de titulares idénticos para no puntuar repetidos.
    uniq = df["Article_title"].drop_duplicates().tolist()
    print(f"titulares únicos a puntuar con FinBERT: {len(uniq)}")

    from transformers import pipeline
    pipe = pipeline("text-classification", model="ProsusAI/finbert", top_k=None,
                    truncation=True, max_length=64)

    net_by_title = {}
    B = 64
    for i in range(0, len(uniq), B):
        batch = uniq[i:i + B]
        outs = pipe(batch, batch_size=B)
        for title, scores in zip(batch, outs):
            p = {d["label"].lower(): float(d["score"]) for d in scores}
            net_by_title[title] = p.get("positive", 0.0) - p.get("negative", 0.0)
        if (i // B) % 20 == 0:
            print(f"  {min(i + B, len(uniq))}/{len(uniq)} titulares puntuados", flush=True)

    df["_net"] = df["Article_title"].map(net_by_title)
    agg = df.groupby("_d").agg(
        sent_net=("_net", "mean"),
        n_titles=("_net", "size"),
    ).reset_index().rename(columns={"_d": "date"})
    agg["date"] = agg["date"].dt.strftime("%Y-%m-%d")
    agg = agg.sort_values("date")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    agg.to_csv(OUT, index=False)
    print(f"\n→ {OUT}  ({len(agg)} días)")
    print(agg.describe(include='all').to_string()[:600])


if __name__ == "__main__":
    main()
