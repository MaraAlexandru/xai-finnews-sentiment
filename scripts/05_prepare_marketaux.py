#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare Marketaux JSON news into:
  1) articles CSV (one row per article)
  2) article–symbol pairs CSV (one row per {article, symbol})
  3) annotation sample CSV (N articles for manual labels)

Notes:
- Handles both a single JSON array file and JSON Lines (one JSON object per line).
- Drops indices/ETFs/FX tickers if requested (heuristics).
- Deduplicates pairs by (article_id, symbol).
- Computes a dominant industry per article (for stratified sampling).

CSV outputs:
  data/processed/marketaux_news_articles.csv
  data/processed/marketaux_news_pairs.csv
  data/annotation/annotation_sample.csv
"""

import argparse
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from datetime import datetime

def read_json_any(path: Path):
    """Read either a JSON array or JSON Lines file. Return list of dicts."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            # JSON array
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            return data
        else:
            # JSON lines
            rows = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    # skip bad line
                    pass
            return rows

def make_article_id(row):
    """Use provided 'uuids' if present; else hash(title|url)."""
    uuids = row.get("uuids") or []
    if isinstance(uuids, list) and len(uuids) > 0:
        return uuids[0]
    base = f"{row.get('title','')}|{row.get('url','')}"
    return hashlib.md5(base.encode("utf-8", errors="ignore")).hexdigest()

def normalize_ts(ts):
    """Normalize published_at; return ISO 8601 or ''."""
    if not ts:
        return ""
    try:
        # most Marketaux timestamps are already ISO-ish
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).isoformat()
    except Exception:
        try:
            return pd.to_datetime(ts, errors="coerce", utc=True).isoformat()
        except Exception:
            return ""

INDEX_KEYWORDS = {"INDEX", "INDICES", "DOW", "S&P", "SECTOR", "COMPOSITE"}
ETF_KEYWORDS = {"ETF", "EXCHANGE TRADED FUND", "FUND", "ISHARES", "VANGUARD", "SPDR", "SELECT SECTOR"}
FX_SUFFIXES = {"USD","EUR","JPY","GBP","CHF","CAD","AUD","NZD","CNY","HKD","SGD","SEK","NOK","DKK","ZAR","MXN","BRL","INR"}

def is_index_or_etf(symbol: str, company_name: str) -> bool:
    """Heuristics to drop indices/ETFs/FX from symbol list."""
    s = (symbol or "").upper()
    name = (company_name or "").upper()

    # indices/benchmarks (e.g., ^SPX, ^NDX, ^HSI)
    if s.startswith("^"):
        return True

    # FX pairs (e.g., EURUSD). Also some appear with slash "EUR/USD"
    if "/" in s:
        return True
    if len(s) == 6 and s.isalpha() and (s[:3] in FX_SUFFIXES and s[3:] in FX_SUFFIXES):
        return True

    # obvious ETF/fund cues in name
    for kw in ETF_KEYWORDS:
        if kw in name:
            return True

    # obvious index cues in name
    for kw in INDEX_KEYWORDS:
        if kw in name:
            return True

    return False

def safe_list_symbols(row):
    """Return list of {symbol, company_name, industry} dicts."""
    syms = row.get("symbols") or []
    if not isinstance(syms, list):
        return []
    out = []
    for s in syms:
        if not isinstance(s, dict):
            continue
        out.append({
            "symbol": s.get("symbol",""),
            "company_name": s.get("company_name",""),
            "industry": s.get("industry","") or ""
        })
    return out

def build_tables(raw, drop_indices_etfs: bool):
    """Create articles table and pairs table."""
    art_rows = []
    pairs = []

    for r in raw:
        aid = make_article_id(r)
        title = r.get("title","").strip()
        desc  = r.get("description","") or ""
        published = normalize_ts(r.get("published_at",""))
        source = r.get("source","") or ""
        url = r.get("url","") or ""

        syms = safe_list_symbols(r)

        # Optionally filter out indices/ETFs/FX
        if drop_indices_etfs:
            syms = [s for s in syms if not is_index_or_etf(s["symbol"], s["company_name"])]

        # dedupe symbols per article
        seen = set()
        clean_syms = []
        for s in syms:
            key = (s["symbol"], s["company_name"], s["industry"])
            if key in seen: 
                continue
            seen.add(key)
            clean_syms.append(s)

        # dominant industry for the article (most frequent non-empty)
        inds = [s["industry"] for s in clean_syms if s["industry"]]
        dom_industry = ""
        if inds:
            dom_industry = Counter(inds).most_common(1)[0][0]

        # stick the compact list of tickers for the article row
        tickers = [s["symbol"] for s in clean_syms]
        industries = sorted({s["industry"] for s in clean_syms if s["industry"]})
        art_rows.append({
            "article_id": aid,
            "published_at": published,
            "source": source,
            "url": url,
            "title": title,
            "description": desc,
            "tickers": ",".join(tickers[:50]),
            "industries": ",".join(industries[:50]),
            "dominant_industry": dom_industry
        })

        # build pair rows
        for s in clean_syms:
            pairs.append({
                "article_id": aid,
                "published_at": published,
                "source": source,
                "url": url,
                "title": title,
                "symbol": s["symbol"],
                "company_name": s["company_name"],
                "industry": s["industry"]
            })

    # Deduplicate pairs just in case (by article_id + symbol)
    df_pairs = pd.DataFrame(pairs)
    if not df_pairs.empty:
        df_pairs["__key"] = df_pairs["article_id"] + "||" + df_pairs["symbol"].astype(str)
        df_pairs = df_pairs.drop_duplicates("__key").drop(columns="__key")

    df_articles = pd.DataFrame(art_rows).drop_duplicates("article_id")
    return df_articles, df_pairs

def stratified_article_sample(df_articles: pd.DataFrame, n: int, seed: int = 42):
    """Stratify by dominant_industry where possible; otherwise random."""
    rng = np.random.default_rng(seed)
    if df_articles.empty:
        return df_articles

    # If no industry info, just random sample
    if df_articles["dominant_industry"].fillna("").eq("").all():
        return df_articles.sample(n=min(n, len(df_articles)), random_state=seed)

    # group by industry and sample proportionally (min 1 per non-empty group)
    groups = [g for _, g in df_articles.groupby(df_articles["dominant_industry"].replace("", np.nan), dropna=False)]
    # split non-empty and empty-industry groups
    non_empty = [g for g in groups if g["dominant_industry"].iloc[0] == g["dominant_industry"].iloc[0]]  # keep all
    empty_grp = df_articles[df_articles["dominant_industry"].replace("", np.nan).isna()]

    # allocate at least 1 per non-empty group, then fill the rest by weight
    k = len(non_empty)
    base_take = min(1, n) if k > 0 else 0
    takes = defaultdict(int)
    remaining = n

    # ensure 1 from each non-empty group if enough articles
    for idx, g in enumerate(non_empty):
        t = min(base_take, len(g))
        takes[idx] += t
        remaining -= t

    if remaining < 0:
        remaining = 0

    # proportional by group sizes for remaining slots
    sizes = np.array([len(g) for g in non_empty], dtype=float)
    if sizes.sum() > 0 and remaining > 0:
        weights = sizes / sizes.sum()
        extra = (weights * remaining).astype(int)
        # fix rounding
        while extra.sum() > remaining:
            i = np.argmax(extra)
            extra[i] -= 1
        while extra.sum() < remaining:
            i = np.argmin(extra)
            extra[i] += 1
        for i, e in enumerate(extra):
            takes[i] += e

    # sample from each non-empty group
    sampled = []
    for i, g in enumerate(non_empty):
        t = min(takes[i], len(g))
        if t > 0:
            sampled.append(g.sample(n=t, random_state=seed))
    sampled_df = pd.concat(sampled) if sampled else pd.DataFrame(columns=df_articles.columns)

    # if still need more, fill from empty industry group
    need = n - len(sampled_df)
    if need > 0 and not empty_grp.empty:
        add = empty_grp.sample(n=min(need, len(empty_grp)), random_state=seed)
        sampled_df = pd.concat([sampled_df, add], ignore_index=True)

    # If still short (few articles overall), just return all
    if len(sampled_df) < n and len(df_articles) > len(sampled_df):
        rest_need = min(n - len(sampled_df), len(df_articles) - len(sampled_df))
        pool = df_articles.loc[~df_articles["article_id"].isin(sampled_df["article_id"])]
        if rest_need > 0 and not pool.empty:
            sampled_df = pd.concat([sampled_df, pool.sample(n=rest_need, random_state=seed)], ignore_index=True)

    return sampled_df.head(n)

def make_annotation_csv(df_articles: pd.DataFrame):
    """Add blank annotation columns."""
    out = df_articles[["article_id","published_at","source","url","title","description","industries","dominant_industry"]].copy()
    out["manual_sentiment"] = ""   # {0,1,2}
    out["manual_rationale"] = ""   # free text
    out["manual_scope"] = ""       # e.g., which company/industry if needed
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to Marketaux JSON (array or JSON Lines).")
    ap.add_argument("--outdir", required=True, help="Output directory for processed CSVs.")
    ap.add_argument("--sample_out", required=True, help="Output directory for annotation sample CSV.")
    ap.add_argument("--sample_size", type=int, default=100, help="Number of articles to include for manual annotation.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop_indices_etfs", type=int, default=1, help="1 to drop indices/ETFs/FX tickers, 0 to keep all.")
    args = ap.parse_args()

    in_path = Path(args.json)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ann_dir = Path(args.sample_out); ann_dir.mkdir(parents=True, exist_ok=True)

    raw = read_json_any(in_path)
    if not isinstance(raw, list) or len(raw) == 0:
        raise SystemExit(f"No records parsed from {in_path}")

    df_articles, df_pairs = build_tables(raw, drop_indices_etfs=bool(args.drop_indices_etfs))

    # Save articles & pairs
    art_csv = outdir / "marketaux_news_articles.csv"
    pairs_csv = outdir / "marketaux_news_pairs.csv"
    df_articles.to_csv(art_csv, index=False)
    df_pairs.to_csv(pairs_csv, index=False)

    # Build annotation sample
    sample_df = stratified_article_sample(df_articles, n=args.sample_size, seed=args.seed)
    sample_csv = ann_dir / "annotation_sample.csv"
    make_annotation_csv(sample_df).to_csv(sample_csv, index=False)

    print(f"Wrote {len(df_articles)} articles -> {art_csv}")
    print(f"Wrote {len(df_pairs)} article–symbol pairs -> {pairs_csv}")
    print(f"Wrote {len(sample_df)}-article annotation sample -> {sample_csv}")

if __name__ == "__main__":
    main()
