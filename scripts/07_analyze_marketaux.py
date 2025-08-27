#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_analyze_marketaux.py
-----------------------

Purpose
=======
1) Load your Marketaux articles CSV (from step 05).
2) (Weak) label each article with VADER compound -> {0=neg,1=neu,2=pos}.
3) Train a transparent baseline (TF-IDF + Logistic Regression) on those weak labels.
4) Report CV/holdout metrics, make SHAP summary, and write a top-token impact table.
5) Optional filtering by industry (sector) and/or date window.

Why this matters
================
- Lets you see what the LR+TFIDF learns from finance news in a given sector/date range.
- Gives interpretable token-level evidence via SHAP.
- Produces paper-ready artifacts (metrics, plots, token tables) for each slice.

Inputs (CSV columns expected)
=============================
- text columns: provide with --text_cols (comma-separated). Typical: "title,description".
- industries column: a semicolon/comma-delimited string of sectors (default column name: "industries").
- published_at: ISO datetime string (e.g., "2025-01-02T22:27:04.000000Z" or "2025-01-02 22:27:04").

Outputs (to ./outputs/)
=======================
- marketaux_vader_labels_{name}.csv        # weak labels + compound + text used
- lr_metrics_marketaux_{name}.csv          # CV means/stds + holdout metrics
- lr_classification_report_marketaux_{name}.txt
- shap_summary_marketaux_{name}.png        # SHAP summary plot
- top_tokens_shap_marketaux_{name}.csv     # top +/- tokens by mean SHAP

Notes
=====
- If NLTK/VADER is missing, the script will auto-install the lexicon (nltk.download at runtime).
- If you prefer strict default VADER thresholds, it uses: <= -0.05 -> 0 (neg), >= 0.05 -> 2 (pos), else 1 (neu).
- You can filter with --include_industries or --exclude_industries (comma-separated, case-insensitive).
- Date filters: --start_date and/or --end_date in YYYY-MM-DD (inclusive).
"""

import argparse, json, re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)
from sklearn.utils import Bunch

# -------------------- VADER (with graceful setup) --------------------
# Try NLTK's VADER first; if unavailable, fall back to the vaderSentiment package.
_VADER_PROVIDER = None

def ensure_vader_ready():
    """
    Returns a SentimentIntensityAnalyzer instance (NLTK or vaderSentiment), or None if neither is available.
    """
    global _VADER_PROVIDER

    # 1) NLTK VADER
    try:
        import nltk
        # Correct import path for NLTK's VADER:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer as _NLTK_SIA
        # Make sure the lexicon is available (no-op if already downloaded)
        nltk.download("vader_lexicon", quiet=True)
        _VADER_PROVIDER = "nltk"
        return _NLTK_SIA()
    except Exception:
        pass

    # 2) vaderSentiment package (doesn't require nltk or a separate download)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VS_SIA
        _VADER_PROVIDER = "vaderSentiment"
        return _VS_SIA()
    except Exception:
        return None



# -------------------- helpers --------------------
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-\']+")

def clean_join_text(row: pd.Series, cols: list[str]) -> str:
    parts = []
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            parts.append(str(row[c]).strip())
    if not parts:
        return ""
    txt = " ".join(parts)
    # light normalization
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def parse_industry_list(val: str) -> list[str]:
    if pd.isna(val):
        return []
    s = str(val)
    # handle comma OR semicolon separated lists
    if ";" in s:
        toks = [t.strip() for t in s.split(";")]
    else:
        toks = [t.strip() for t in s.split(",")]
    # drop empties
    return [t for t in toks if t]

def in_any_industry(row_inds: list[str], inc: set[str]) -> bool:
    # case-insensitive matching
    rset = {x.lower() for x in row_inds}
    return any(x in rset for x in inc)

def macro_roc_auc_ovr(y_true, proba):
    """One-vs-rest macro AUC, safe for binary or multiclass."""
    classes = np.unique(y_true)
    aucs = []
    for k in classes:
        try:
            k = int(k)
        except Exception:
            continue
        if k < 0 or k >= proba.shape[1]:
            continue
        try:
            aucs.append(roc_auc_score((y_true == k).astype(int), proba[:, k]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else float("nan")

def to_float(x):
    try:
        return float(np.asarray(x).item())
    except Exception:
        return float(x)

def build_outputs_name(csv_path: str, tag: str) -> str:
    """e.g., 'marketaux_articles' + tag -> 'marketaux_articles_tag'"""
    stem = Path(csv_path).stem
    return f"{stem}_{tag}" if tag else stem


# -------------------- core pipeline --------------------
def run_pipeline(args: argparse.Namespace) -> Bunch:
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)
    name_tag = []
    if args.include_industries:
        name_tag.append("inc" + "-".join(sorted([s.strip().lower() for s in args.include_industries.split(",") if s.strip()])))
    if args.exclude_industries:
        name_tag.append("exc" + "-".join(sorted([s.strip().lower() for s in args.exclude_industries.split(",") if s.strip()])))
    if args.start_date or args.end_date:
        name_tag.append(f"{args.start_date or 'min'}_{args.end_date or 'max'}")
    tag = "_".join(name_tag)
    out_name = build_outputs_name(args.csv, tag)

    # ---------- Load ----------
    df = pd.read_csv(args.csv)
    # make sure columns exist
    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    for c in text_cols:
        if c not in df.columns:
            raise ValueError(f"Missing text column '{c}' in {args.csv}")
    # try to find industries column
    industries_col = args.industries_col
    if industries_col not in df.columns:
        # try a few common alternatives
        candidates = [c for c in df.columns if c.lower() in {"industries","industry","sectors","sector_list"}]
        if candidates:
            industries_col = candidates[0]
        else:
            industries_col = None
    # try to find datetime column
    time_col = args.time_col if args.time_col in df.columns else None
    if time_col is None:
        for c in df.columns:
            if c.lower() in {"published_at","date","datetime","time"}:
                time_col = c; break

    # ---------- Filter by date ----------
    if time_col:
        # parse very leniently
        def _parse_dt(x):
            try:
                return datetime.fromisoformat(str(x).replace("Z", "+00:00"))
            except Exception:
                # try fallback without timezone
                try:
                    return datetime.fromisoformat(str(x).split("Z")[0])
                except Exception:
                    return None
        dt = df[time_col].apply(_parse_dt)
        df = df.loc[dt.notna()].copy()
        dt = dt.loc[df.index]
        if args.start_date:
            start_dt = datetime.fromisoformat(args.start_date)
            df = df.loc[dt >= start_dt]
        if args.end_date:
            end_dt = datetime.fromisoformat(args.end_date)  # inclusive end-of-day handled by user
            df = df.loc[dt <= end_dt]

    # ---------- Filter by industries ----------
    if industries_col and (args.include_industries or args.exclude_industries):
        inc = set([s.strip().lower() for s in (args.include_industries or "").split(",") if s.strip()])
        exc = set([s.strip().lower() for s in (args.exclude_industries or "").split(",") if s.strip()])
        inds = df[industries_col].apply(parse_industry_list)
        mask = pd.Series(True, index=df.index)
        if inc:
            mask &= inds.apply(lambda xs: in_any_industry(xs, inc))
        if exc:
            mask &= ~inds.apply(lambda xs: in_any_industry(xs, exc))
        df = df.loc[mask].copy()

    # ---------- Build text, basic length filters ----------
    df["__text__"] = df.apply(lambda r: clean_join_text(r, text_cols), axis=1)
    if args.min_chars > 0:
        df = df.loc[df["__text__"].str.len() >= args.min_chars]
    if args.min_words > 0:
        df = df.loc[df["__text__"].str.split().apply(len) >= args.min_words]
    df = df.reset_index(drop=True)

    if len(df) < 30:
        raise ValueError(f"Too few rows ({len(df)}) after filtering; relax filters or widen date/industry.")

    print(f"Loaded {len(df)} rows from {args.csv} after filters. Text columns used: {text_cols}")

    # ---------- VADER weak labels ----------
    sia = ensure_vader_ready()
    if sia is None:
        raise RuntimeError("VADER (nltk) is not available. Please: pip install nltk && python -c \"import nltk; nltk.download('vader_lexicon')\"")

    compounds = df["__text__"].apply(lambda t: sia.polarity_scores(str(t))["compound"]).astype(np.float32).values
    # map to tri-class labels
    thr_neg, thr_pos = -0.05, 0.05
    y = np.where(compounds <= thr_neg, 0, np.where(compounds >= thr_pos, 2, 1)).astype(int)

    # optional rebalance report
    cls, counts = np.unique(y, return_counts=True)
    dist = {int(c): int(n) for c, n in zip(cls, counts)}
    print(f"VADER label distribution: {dist}")

    # write weakly-labeled CSV (for reproducibility)
    keep_cols = []
    for c in ["article_id","published_at","source","url"] + text_cols:
        if c in df.columns:
            keep_cols.append(c)
    labeled = pd.DataFrame({
        **{c: df[c].values for c in keep_cols},
        "compound": compounds,
        "y": y
    })
    labeled.to_csv(Path("outputs") / f"marketaux_vader_labels_{out_name}.csv", index=False)

    # ---------- Split ----------
    Xtr_txt, Xte_txt, ytr, yte = train_test_split(
        df["__text__"], y, test_size=0.2, stratify=y, random_state=args.seed
    )

    # ---------- Vectorize ----------
    vec = TfidfVectorizer(
        ngram_range=(1, args.max_ngram),
        min_df=args.min_df,
        max_df=args.max_df,
        stop_words="english",
        dtype=np.float32
    )
    Xtr = vec.fit_transform(Xtr_txt)
    Xte = vec.transform(Xte_txt)

    # ---------- Model ----------
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, multi_class="auto")

    cv = cross_validate(
        lr, Xtr, ytr, cv=5, n_jobs=-1, scoring={"acc":"accuracy","f1":"f1_macro"}
    )
    cv_acc_mean = to_float(np.mean(cv["test_acc"]))
    cv_acc_std  = to_float(np.std(cv["test_acc"]))
    cv_f1_mean  = to_float(np.mean(cv["test_f1"]))
    cv_f1_std   = to_float(np.std(cv["test_f1"]))
    print(f"CV: acc={cv_acc_mean:.3f}±{cv_acc_std:.3f}  f1={cv_f1_mean:.3f}±{cv_f1_std:.3f}")

    lr.fit(Xtr, ytr)
    proba = lr.predict_proba(Xte)
    pred  = proba.argmax(1)

    acc = to_float(accuracy_score(yte, pred))
    f1  = to_float(f1_score(yte, pred, average="macro"))
    auc = to_float(macro_roc_auc_ovr(yte, proba))
    print(f"Holdout: acc={acc:.3f}  macro-f1={f1:.3f}  macro-auc={auc:.3f}")

    # ---------- Save metrics ----------
    out = Path("outputs")
    pd.DataFrame([
        {"metric":"cv_acc_mean","value":cv_acc_mean},
        {"metric":"cv_acc_std","value":cv_acc_std},
        {"metric":"cv_f1_mean","value":cv_f1_mean},
        {"metric":"cv_f1_std","value":cv_f1_std},
        {"metric":"holdout_acc","value":acc},
        {"metric":"holdout_macro_f1","value":f1},
        {"metric":"holdout_macro_auc","value":auc},
        {"metric":"n_train","value":int(Xtr.shape[0])},
        {"metric":"n_test","value":int(Xte.shape[0])}
    ]).to_csv(out / f"lr_metrics_marketaux_{out_name}.csv", index=False)

    with open(out / f"lr_classification_report_marketaux_{out_name}.txt","w", encoding="utf-8") as f:
        f.write(classification_report(yte, pred, digits=3))

    # ---------- SHAP ----------
    try:
        import shap
        # background sample (fast & stable)
        background = shap.sample(Xtr, min(200, Xtr.shape[0]), random_state=args.seed)
        explainer  = shap.LinearExplainer(lr, background, feature_names=vec.get_feature_names_out())
        shap_vals  = explainer.shap_values(Xte)

        # summary plot
        plt.figure()
        shap.summary_plot(shap_vals, Xte, feature_names=vec.get_feature_names_out(), show=False)
        plt.tight_layout()
        plt.savefig(out / f"shap_summary_marketaux_{out_name}.png", dpi=250)
        plt.close()

        # token impact table
        def _to_ndarray_2d(x):
            if hasattr(x, "toarray"):
                x = x.toarray()
            elif isinstance(x, (list, tuple)):
                arrs = [a.toarray() if hasattr(a, "toarray") else np.asarray(a) for a in x]
                x = np.mean(np.stack(arrs, axis=0), axis=0)  # (N,F)
            return np.asarray(x)

        sv = _to_ndarray_2d(shap_vals)  # (n_samples,n_features)
        names = np.array(vec.get_feature_names_out(), dtype=object)
        pos_mean = np.asarray(np.mean(np.where(sv > 0, sv, 0), axis=0)).ravel()
        neg_mean = np.asarray(np.mean(np.where(sv < 0, sv, 0), axis=0)).ravel()

        K = max(5, int(args.top_k_tokens))
        pos_idx = np.argsort(-pos_mean)[:K]
        neg_idx = np.argsort(neg_mean)[:K]

        token_df = pd.DataFrame({
            "Positive token": names[pos_idx].tolist(),
            "Mean SHAP (+)": pos_mean[pos_idx].astype(float).tolist(),
            "Negative token": names[neg_idx].tolist(),
            "Mean SHAP (–)": neg_mean[neg_idx].astype(float).tolist(),
        })
        token_df.to_csv(out / f"top_tokens_shap_marketaux_{out_name}.csv", index=False)
    except Exception as e:
        print(f"[SHAP] Skipped due to: {repr(e)}")

    # done
    return Bunch(
        n_rows=len(df),
        label_dist=dist,
        out_name=out_name
    )


# -------------------- CLI --------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to Marketaux articles CSV (from step 05).")
    ap.add_argument("--text_cols", default="title,description", help="Comma-separated text columns to concatenate.")
    ap.add_argument("--industries_col", default="industries", help="Column name with industries list.")
    ap.add_argument("--time_col", default="published_at", help="Datetime column name (ISO).")
    ap.add_argument("--include_industries", default="", help="Comma-separated industries to INCLUDE (case-insensitive).")
    ap.add_argument("--exclude_industries", default="", help="Comma-separated industries to EXCLUDE (case-insensitive).")
    ap.add_argument("--start_date", default="", help="Filter start date (YYYY-MM-DD), inclusive.")
    ap.add_argument("--end_date", default="", help="Filter end date (YYYY-MM-DD), inclusive.")

    ap.add_argument("--min_chars", type=int, default=15, help="Drop rows with too-short text.")
    ap.add_argument("--min_words", type=int, default=3, help="Drop rows with too few words.")

    ap.add_argument("--max_ngram", type=int, default=2, help="Use 1..N n-grams for TF-IDF.")
    ap.add_argument("--min_df", type=int, default=3, help="TF-IDF min_df.")
    ap.add_argument("--max_df", type=float, default=0.95, help="TF-IDF max_df.")

    ap.add_argument("--top_k_tokens", type=int, default=20, help="How many top +/- tokens to export from SHAP.")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    run_pipeline(args)
