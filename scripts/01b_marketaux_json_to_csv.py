#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01b_marketaux_json_to_csv.py
----------------------------

Purpose
=======
Ingest Marketaux-style JSON news into tidy CSVs for your NLP pipeline.

Input formats supported:
- A JSON *list* of article dicts (your example)
- JSONL (one JSON object per line)

Output (written to --out_dir):
1) {base}_articles.csv
   One row per article with aggregated symbols/industries and an empty 'y' label.
   Columns include: id, title, description, text, y, published_at, source, url,
                    n_symbols, symbols, companies, industries, industries_primary

2) {base}_article_symbols.csv
   One row per (article × symbol) with symbol/company/industry and a guessed
   instrument_type (equity/etf/index/fx/other). Useful for sector filtering.

3) {base}_annotation.csv
   Minimal sheet (id, text, y, url, source, published_at, symbols, industries)
   ready for manual labeling (0=neg, 1=neu, 2=pos).

Why three files?
- Articles: analytics-ready rollup
- Exploded: sector/company joins
- Annotation: exactly matches your downstream scripts' expected schema (text,y)

Usage
=====
# Typical
python scripts/01b_marketaux_json_to_csv.py ^
  --in data/raw/marketaux/news.json ^
  --out_dir data/processed/marketaux ^
  --base jan02_2025

# JSONL
python scripts/01b_marketaux_json_to_csv.py --in data/raw/marketaux/news.jsonl --out_dir data/processed/marketaux

Notes
=====
- We keep label column 'y' BLANK; fill 0/1/2 manually when you annotate.
- Duplicate URLs are dropped (keep first).
- 'id' prefers the first 'uuid' provided; otherwise a short SHA1 of the URL.
- 'text' = title + " — " + description (description optional).
- We infer a simple instrument_type per symbol: equity / etf / index / fx / other.
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Iterable

import pandas as pd


def read_json_any(path: Path) -> List[Dict[str, Any]]:
    """
    Read a Marketaux-like file that may be:
      - a JSON list of dicts
      - a JSONL (one JSON object per line)
    Returns a list of dicts (possibly empty).
    """
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return []
    # JSON list?
    if txt[0] == "[":
        return json.loads(txt)
    # JSONL fallback
    items = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            # skip malformed lines
            pass
    return items


def sha1_short(s: str, n: int = 12) -> str:
    """Short, stable identifier for fallback (e.g., when uuids are missing)."""
    h = hashlib.sha1(str(s).encode("utf-8", errors="ignore")).hexdigest()
    return h[:n]


def norm_list_str(values: Iterable[str], sep: str = ", "):
    """
    Join a list into a readable, de-duplicated string (order-preserving).
    - Strips whitespace, ignores empties/None.
    """
    seen = set()
    out = []
    for v in values:
        v = (v or "").strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return sep.join(out)


def pick_primary_industry(inds: List[str]) -> str:
    """
    Pick a 'primary' industry for convenience:
    - first non-empty industry in original order, else 'Unknown'
    """
    for v in inds:
        v = (v or "").strip()
        if v:
            return v
    return "Unknown"


def infer_instrument_type(symbol: str, company_name: str) -> str:
    """
    Lightweight guess of instrument type based on symbol/name patterns.
    It's just to help you filter (not a ground-truth taxonomy).
    """
    s = (symbol or "").upper()
    name = (company_name or "").upper()

    if "/" in s:              # e.g., EUR/USD
        return "fx"
    if s.startswith("^"):     # vendor aliases/indices in many feeds
        return "index"
    if "ETF" in name:
        return "etf"
    # crude index / composite flags
    if any(k in name for k in ["INDEX", "COMPOSITE", "SECTOR", "SELECT SECTOR"]):
        return "index"
    # everything else: assume equity by default
    if s:
        return "equity"
    return "other"


def build_rows(articles: List[Dict[str, Any]]):
    """
    Transform raw JSON items into:
      - articles_rows: one row per article with aggregated lists
      - exploded_rows: one row per (article × symbol)
    We drop duplicate URLs (keep first occurrence).
    """
    seen_urls = set()
    articles_rows = []
    exploded_rows = []

    for item in articles:
        title = (item.get("title") or "").strip()
        desc  = (item.get("description") or "").strip()
        url   = (item.get("url") or "").strip()
        if not url:
            # skip articles without a URL anchor
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)

        published_at = (item.get("published_at") or "").strip()
        source       = (item.get("source") or "").strip()
        uuids        = item.get("uuids") or []
        if isinstance(uuids, list) and uuids:
            _id = str(uuids[0])
        else:
            _id = sha1_short(url)

        # symbols list: each a dict with symbol/company_name/industry
        syms = item.get("symbols") or []
        sym_list = []
        comp_list = []
        ind_list = []

        # also populate exploded
        for s in syms:
            sym = (s.get("symbol") or "").strip()
            comp = (s.get("company_name") or "").strip()
            ind  = (s.get("industry") or "").strip()

            if sym or comp or ind:
                sym_list.append(sym)
                comp_list.append(comp)
                ind_list.append(ind)

                exploded_rows.append({
                    "id": _id,
                    "url": url,
                    "published_at": published_at,
                    "source": source,
                    "title": title,
                    "description": desc,
                    "symbol": sym,
                    "company_name": comp,
                    "industry": ind,
                    "instrument_type": infer_instrument_type(sym, comp),
                })

        # aggregate deduped strings for article-level view
        symbols_str    = norm_list_str(sym_list)
        companies_str  = norm_list_str(comp_list)
        industries_str = norm_list_str(ind_list)
        n_symbols      = len({s for s in sym_list if (s or "").strip()})

        # concatenated text for NLP (title + description if present)
        if title and desc:
            text = f"{title} — {desc}"
        else:
            text = title or desc

        articles_rows.append({
            "id": _id,
            "title": title,
            "description": desc,
            "text": text,
            "y": "",  # keep empty for manual labeling (0=neg, 1=neu, 2=pos)
            "published_at": published_at,
            "source": source,
            "url": url,
            "n_symbols": n_symbols,
            "symbols": symbols_str,
            "companies": companies_str,
            "industries": industries_str,
            "industries_primary": pick_primary_industry(ind_list),
        })

    return articles_rows, exploded_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Path to Marketaux JSON (.json list) or JSONL (.jsonl) file")
    ap.add_argument("--out_dir", default="data/processed/marketaux",
                    help="Directory to write CSV outputs")
    ap.add_argument("--base", default="marketaux",
                    help="Base filename (prefix) for outputs")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    articles = read_json_any(in_path)
    if not articles:
        print(f"[WARN] No articles found in {in_path}")
        return

    art_rows, expl_rows = build_rows(articles)

    # DataFrames
    df_articles = pd.DataFrame(art_rows)
    df_exploded = pd.DataFrame(expl_rows)

    # Sort for readability
    if not df_articles.empty:
        df_articles = df_articles.sort_values(by=["published_at", "source"], ascending=[False, True])
    if not df_exploded.empty:
        df_exploded = df_exploded.sort_values(by=["published_at", "source", "symbol"], ascending=[False, True, True])

    # File paths
    f_articles = out_dir / f"{args.base}_articles.csv"
    f_exploded = out_dir / f"{args.base}_article_symbols.csv"
    f_annot    = out_dir / f"{args.base}_annotation.csv"

    # Save with UTF-8 BOM so Excel opens cleanly on Windows
    df_articles.to_csv(f_articles, index=False, encoding="utf-8-sig")
    df_exploded.to_csv(f_exploded, index=False, encoding="utf-8-sig")

    # Minimal annotation sheet (exactly what your training scripts like)
    annot_cols = ["id", "text", "y", "url", "source", "published_at", "symbols", "industries"]
    df_articles[annot_cols].to_csv(f_annot, index=False, encoding="utf-8-sig")

    print(f"[OK] Wrote:\n  {f_articles}\n  {f_exploded}\n  {f_annot}")
    print("\nNext steps:")
    print("- Open the annotation CSV and fill the 'y' column with 0/1/2.")
    print("- Then you can run your baselines on it, e.g.:")
    print("  python scripts/02_run_lr_shap.py --csv data/processed/marketaux/marketaux_annotation.csv")
    print("  python scripts/03_run_finbert.py --csv data/processed/marketaux/marketaux_annotation.csv --mode inference")
    print("  (for finetune, you need labeled y values first)")


if __name__ == "__main__":
    main()
