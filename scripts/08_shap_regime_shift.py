#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08_shap_regime_shift.py

Compares Q1 vs Q2 token importance using SHAP for an LR+TFIDF model trained on
VADER weak labels. Exports top tokens per quarter and a rank-shift table to
highlight "regime changes" in language.
"""

import argparse
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for saving figures
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------- VADER (robust import) --------------------
def ensure_vader_ready():
    """
    Return a SentimentIntensityAnalyzer instance by trying NLTK first,
    then falling back to the vaderSentiment package if available.
    """
    # Attempt to use NLTK's VADER implementation
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer as _NLTK_SIA
        nltk.download("vader_lexicon", quiet=True)
        return _NLTK_SIA()
    except Exception:
        pass
    # Fallback to the standalone vaderSentiment package
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VS_SIA
        return _VS_SIA()
    except Exception:
        return None

# -------------------- Helpers --------------------
def clean_join_text(row: pd.Series, cols):
    """Join selected text columns and lightly normalize whitespace."""
    parts = []
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            parts.append(str(row[c]).strip())
    if not parts:
        return ""
    txt = " ".join(parts)
    return re.sub(r"\s+", " ", txt).strip()

def parse_industry_list(val: str):
    """Parse a cell that may be comma- or semicolon-delimited."""
    if pd.isna(val):
        return []
    s = str(val)
    toks = [t.strip() for t in re.split(r"[;,]", s)]
    return [t for t in toks if t]

def in_any_industry(row_inds, inc_set):
    """Case-insensitive check for industry membership."""
    rset = {x.lower() for x in row_inds}
    return any(x in rset for x in inc_set)

def label_with_vader(texts, thr_neg=-0.05, thr_pos=0.05):
    """Weak-label texts using VADER compound scores."""
    sia = ensure_vader_ready()
    if sia is None:
        raise RuntimeError("VADER not available. Install with: pip install nltk vaderSentiment")
    compounds = np.array([sia.polarity_scores(str(t))["compound"] for t in texts], dtype=np.float32)
    y = np.where(compounds <= thr_neg, 0, np.where(compounds >= thr_pos, 2, 1)).astype(int)
    return compounds, y

def fit_lr_tfidf(texts, labels, ngram_max=2, min_df=3, max_df=0.95, seed=42):
    """Fit a TF-IDF + Logistic Regression model."""
    vec = TfidfVectorizer(
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
        dtype=np.float32
    )
    X = vec.fit_transform(texts)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, random_state=seed)
    lr.fit(X, labels)
    return vec, lr, X

# -------------------- SHAP Importance Calculation --------------------
def _shap_to_dense_2d(sv):
    """Normalize SHAP outputs to a dense (N,F) ndarray for consistent processing."""
    if hasattr(sv, "toarray"):  # sparse matrix
        sv = sv.toarray()
    elif isinstance(sv, (list, tuple)): # multiclass output
        arrs = [a.toarray() if hasattr(a, "toarray") else np.asarray(a) for a in sv]
        sv = np.mean(np.stack(arrs, axis=0), axis=0) # Average over classes
    sv = np.asarray(sv)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    return sv

def compute_token_importances(lr, X, feature_names, seed=42):
    """
    Compute per-feature impact using SHAP, with a coefficient-based fallback.
    Returns aligned feature names and their positive/negative impact scores.
    """
    names = np.array(feature_names, dtype=object)
    try:
        import shap
        background = shap.sample(X, min(200, X.shape[0]), random_state=seed)
        explainer  = shap.LinearExplainer(lr, background, feature_names=names)
        sv = _shap_to_dense_2d(explainer.shap_values(X))
        
        # Align feature names and SHAP values in case of length mismatch
        F_names, F_shap = names.shape[0], sv.shape[1]
        F = min(F_names, F_shap)
        if F_names != F_shap:
            print(f"[SHAP] Warning: feature length mismatch (names={F_names}, shap={F_shap}); aligning to {F}.")
        names = names[:F]
        sv    = sv[:, :F]
        
        pos_mean = np.mean(np.where(sv > 0, sv, 0), axis=0).ravel()
        neg_mean = np.mean(np.where(sv < 0, sv, 0), axis=0).ravel()
        return names, pos_mean, neg_mean, "shap"
    except Exception:
        # Fallback proxy: coefficient * mean TF-IDF feature value
        print("[SHAP] SHAP failed, using coefficient proxy for feature importance.")
        X_mean = np.asarray(X.mean(axis=0)).ravel()
        coefs = lr.coef_
        cneg = 0
        cpos = 2 if coefs.shape[0] >= 3 else (coefs.shape[0] - 1)
        
        pos_imp = np.maximum(0.0, coefs[cpos]) * X_mean
        neg_imp = np.minimum(0.0, coefs[cneg]) * X_mean
        return names, pos_imp.astype(float), neg_imp.astype(float), "coef_proxy"

def top_tokens_table(feature_names, pos_mean, neg_mean, k=25, window_label="Q1"):
    """Build a DataFrame of top-K positive & negative tokens by mean impact."""
    names = np.asarray(feature_names, dtype=object)
    pos_mean = np.asarray(pos_mean).ravel()
    neg_mean = np.asarray(neg_mean).ravel()

    L = min(len(names), len(pos_mean), len(neg_mean))
    if L == 0:
        return pd.DataFrame(columns=["token","direction","mean_impact","rank","window"])
    
    names, pos_mean, neg_mean = names[:L], pos_mean[:L], neg_mean[:L]
    k = int(min(max(1, k), L))
    pos_idx = np.argsort(-pos_mean)[:k]
    neg_idx = np.argsort(neg_mean)[:k]

    df_pos = pd.DataFrame({
        "token": names[pos_idx], "direction": "pos", "mean_impact": pos_mean[pos_idx],
        "rank": np.arange(1, len(pos_idx)+1), "window": window_label
    })
    df_neg = pd.DataFrame({
        "token": names[neg_idx], "direction": "neg", "mean_impact": neg_mean[neg_idx],
        "rank": np.arange(1, len(neg_idx)+1), "window": window_label
    })
    return pd.concat([df_pos, df_neg], ignore_index=True)

def build_rank_shift(q1_df, q2_df, k=25):
    """Join Q1 vs Q2 top lists and compute rank deltas."""
    merged = q1_df.merge(q2_df, on=["token", "direction"], how="outer", suffixes=("_Q1", "_Q2"))
    merged["rank_Q1"] = merged["rank_Q1"].fillna(k + 1).astype(int)
    merged["rank_Q2"] = merged["rank_Q2"].fillna(k + 1).astype(int)
    merged["delta_rank"] = merged["rank_Q1"] - merged["rank_Q2"]
    merged = merged.sort_values(by=["direction", "delta_rank"], ascending=[True, True]).reset_index(drop=True)
    return merged

# -------------------- Data Filtering --------------------
def filter_slice(df, time_col, start_date, end_date, industries_col=None, include_inds=None, min_chars=15, min_words=3, text_cols=("title","description")):
    """Apply time, industry, and text length filters to a DataFrame."""
    tmp = df.copy()

    # Time filtering (robust to timezone formats)
    if time_col and time_col in tmp.columns:
        dt = pd.to_datetime(tmp[time_col], utc=True, errors="coerce")
        tmp = tmp.loc[dt.notna()].copy()
        dt = dt.loc[tmp.index]
        if start_date:
            tmp = tmp.loc[dt >= pd.to_datetime(start_date, utc=True)]
        if end_date:
            tmp = tmp.loc[dt <= pd.to_datetime(end_date, utc=True)]

    # Industry filter
    if industries_col and include_inds:
        inc = {s.strip().lower() for s in include_inds.split(",") if s.strip()}
        inds = tmp[industries_col].apply(parse_industry_list) if industries_col in tmp.columns else pd.Series([[]]*len(tmp), index=tmp.index)
        tmp = tmp.loc[inds.apply(lambda xs: in_any_industry(xs, inc))]

    # Build text & length filters
    tmp["__text__"] = tmp.apply(lambda r: clean_join_text(r, list(text_cols)), axis=1)
    if min_chars > 0:
        tmp = tmp.loc[tmp["__text__"].str.len() >= min_chars]
    if min_words > 0:
        tmp = tmp.loc[tmp["__text__"].str.split().apply(len) >= min_words]
    
    return tmp.reset_index(drop=True)

# -------------------- Main Execution --------------------
def main(args):
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    stem = Path(args.csv).stem

    # Resolve date windows for Q1 and Q2
    if args.q1_start and args.q1_end and args.q2_start and args.q2_end:
        q1_start, q1_end = args.q1_start, args.q1_end
        q2_start, q2_end = args.q2_start, args.q2_end
        tag_dates = f"{q1_start}_{q1_end}_vs_{q2_start}_{q2_end}"
    else:
        yr = int(args.year)
        q1_start, q1_end = f"{yr}-01-01", f"{yr}-03-31"
        q2_start, q2_end = f"{yr}-04-01", f"{yr}-06-30"
        tag_dates = f"Q1vsQ2_{yr}"

    df = pd.read_csv(args.csv)
    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    time_col = args.time_col if args.time_col in df.columns else None
    industries_col = args.industries_col if args.industries_col in df.columns else None

    # Create data slices for each quarter
    q1_df = filter_slice(df, time_col, q1_start, q1_end, industries_col, args.include_industries, args.min_chars, args.min_words, text_cols)
    q2_df = filter_slice(df, time_col, q2_start, q2_end, industries_col, args.include_industries, args.min_chars, args.min_words, text_cols)

    if len(q1_df) < 40 or len(q2_df) < 40:
        raise ValueError(f"Too few rows after filtering: Q1={len(q1_df)}, Q2={len(q2_df)}. Relax filters.")
    print(f"Q1 rows: {len(q1_df)} | Q2 rows: {len(q2_df)}")

    # VADER weak labels per slice
    _, q1_y = label_with_vader(q1_df["__text__"].values, args.vader_neg_thr, args.vader_pos_thr)
    _, q2_y = label_with_vader(q2_df["__text__"].values, args.vader_neg_thr, args.vader_pos_thr)

    # Fit a model for each time window
    vec1, lr1, X1 = fit_lr_tfidf(q1_df["__text__"].values, q1_y, args.max_ngram, args.min_df, args.max_df, args.seed)
    vec2, lr2, X2 = fit_lr_tfidf(q2_df["__text__"].values, q2_y, args.max_ngram, args.min_df, args.max_df, args.seed)

    # Compute token importances for each model
    names1, pos1, neg1, src1 = compute_token_importances(lr1, X1, vec1.get_feature_names_out(), args.seed)
    names2, pos2, neg2, src2 = compute_token_importances(lr2, X2, vec2.get_feature_names_out(), args.seed)

    # Generate top-K tables
    topK = int(args.top_k)
    q1_tok = top_tokens_table(names1, pos1, neg1, k=topK, window_label="Q1")
    q2_tok = top_tokens_table(names2, pos2, neg2, k=topK, window_label="Q2")

    # Create a unique tag for the output files
    tag = f"{stem}_{tag_dates}"
    if args.include_industries:
        tag += "__" + "_".join(sorted([s.strip().lower() for s in args.include_industries.split(",") if s.strip()]))

    # Save per-window tables and the rank shift comparison
    q1_tok.to_csv(out_dir / f"regime_tokens_Q1_{tag}.csv", index=False)
    q2_tok.to_csv(out_dir / f"regime_tokens_Q2_{tag}.csv", index=False)
    
    shift = build_rank_shift(q1_tok, q2_tok, k=topK)
    shift.to_csv(out_dir / f"regime_rank_shift_{tag}.csv", index=False)

    # Plot the top 20 largest rank shifts
    shift_to_plot = shift.reindex(shift["delta_rank"].abs().sort_values(ascending=False).index).head(20)
    if not shift_to_plot.empty:
        plt.figure(figsize=(10, 8))
        colors = ['#2ca02c' if x > 0 else '#d62728' for x in shift_to_plot['delta_rank']]
        ylab = [f"{t} ({d})" for t, d in zip(shift_to_plot["token"], shift_to_plot["direction"])]
        plt.barh(ylab, shift_to_plot["delta_rank"], color=colors)
        plt.axvline(0, color='black', lw=0.8)
        plt.xlabel("Rank Change (Q1 Rank - Q2 Rank) | Negative value means token became MORE important in Q2")
        plt.title(f"Top 20 Token Importance Shifts (Q1 vs Q2 {args.year})")
        plt.gca().invert_yaxis() # Puts the biggest mover at the top
        plt.tight_layout()
        plt.savefig(out_dir / f"regime_shift_top20_{tag}.png", dpi=300)
        plt.close()

    print(f"Saved: regime_tokens_Q1_{tag}.csv, regime_tokens_Q2_{tag}.csv, regime_rank_shift_{tag}.csv, and regime_shift_top20_{tag}.png")
    print(f"Importance source: Q1={src1}, Q2={src2}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to Marketaux articles CSV.")
    ap.add_argument("--text_cols", default="title,description", help="Comma-separated text columns to join.")
    ap.add_argument("--time_col", default="published_at", help="Datetime column (ISO).")
    ap.add_argument("--industries_col", default="industries", help="Column with industries list.")
    ap.add_argument("--include_industries", default="", help="Comma-separated industries to INCLUDE (case-insensitive).")
    ap.add_argument("--min_chars", type=int, default=15, help="Drop rows with very short joined text.")
    ap.add_argument("--min_words", type=int, default=3, help="Drop rows with too few words.")
    ap.add_argument("--year", type=int, default=2025, help="If custom dates not provided, use Q1/Q2 of this year.")
    ap.add_argument("--q1_start", default="", help="YYYY-MM-DD")
    ap.add_argument("--q1_end",   default="", help="YYYY-MM-DD")
    ap.add_argument("--q2_start", default="", help="YYYY-MM-DD")
    ap.add_argument("--q2_end",   default="", help="YYYY-MM-DD")
    ap.add_argument("--max_ngram", type=int, default=2, help="Use 1..N n-grams.")
    ap.add_argument("--min_df", type=int, default=3, help="TF-IDF min_df.")
    ap.add_argument("--max_df", type=float, default=0.95, help="TF-IDF max_df.")
    ap.add_argument("--vader_neg_thr", type=float, default=-0.05, help="VADER threshold for negative.")
    ap.add_argument("--vader_pos_thr", type=float, default=0.05, help="VADER threshold for positive.")
    ap.add_argument("--top_k", type=int, default=25, help="Top-K tokens per side to export.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)