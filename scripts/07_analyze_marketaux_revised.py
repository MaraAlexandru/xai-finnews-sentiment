#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_analyze_marketaux_revised.py
-------------------------------

Purpose:
1) Load your Marketaux articles CSV (from step 05).
2) (Weak) label each article with VADER compound -> {0=neg,1=neu,2=pos}.
3) Train a transparent baseline (TF-IDF + Logistic Regression) on those weak labels.
4) Report CV/holdout metrics, make SHAP summary, and write a top-token impact table.
5) Optional filtering by industry (sector) and/or date window.

REVISIONS:
- This is the complete script, restoring all original functionalities.
- Corrected the SHAP plotting logic to robustly handle multiclass outputs and avoid the ValueError.
- The summary_plot function is now called in a way that correctly handles the list of arrays
  returned by the explainer for multiclass models.
"""

import argparse, json, re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Use a non-interactive backend for saving figures
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)
from sklearn.utils import Bunch

# -------------------- VADER (with graceful setup) --------------------
_VADER_PROVIDER = None

def ensure_vader_ready():
    """
    Returns a SentimentIntensityAnalyzer instance by trying NLTK first,
    then falling back to the vaderSentiment package if available.
    """
    global _VADER_PROVIDER
    # Attempt to use NLTK's VADER implementation
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer as _NLTK_SIA
        nltk.download("vader_lexicon", quiet=True)
        _VADER_PROVIDER = "nltk"
        return _NLTK_SIA()
    except Exception:
        pass

    # Fallback to the standalone vaderSentiment package
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VS_SIA
        _VADER_PROVIDER = "vaderSentiment"
        return _VS_SIA()
    except Exception:
        return None

# -------------------- Helper Functions --------------------
def clean_join_text(row: pd.Series, cols: list[str]) -> str:
    """Concatenates text from specified columns in a row into a single string."""
    parts = []
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            parts.append(str(row[c]).strip())
    if not parts:
        return ""
    txt = " ".join(parts)
    # Normalize whitespace
    return re.sub(r"\s+", " ", txt).strip()

def parse_industry_list(val: str) -> list[str]:
    """Parses a string of industries that may be comma or semicolon delimited."""
    if pd.isna(val):
        return []
    s = str(val)
    toks = [t.strip() for t in re.split(r"[;,]", s)]
    return [t for t in toks if t]

def in_any_industry(row_inds: list[str], inc: set[str]) -> bool:
    """Performs a case-insensitive check for industry membership."""
    rset = {x.lower() for x in row_inds}
    return any(x in rset for x in inc)

def macro_roc_auc_ovr(y_true, proba):
    """Calculates a one-vs-rest macro-averaged AUC score, safe for multiclass cases."""
    classes = np.unique(y_true)
    aucs = []
    for k in classes:
        try:
            k = int(k)
            if k < 0 or k >= proba.shape[1]:
                continue
            aucs.append(roc_auc_score((y_true == k).astype(int), proba[:, k]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else float("nan")

def to_float(x):
    """Safely converts numpy types to standard Python floats."""
    try:
        return float(np.asarray(x).item())
    except Exception:
        return float(x)

def build_outputs_name(csv_path: str, tag: str) -> str:
    """Constructs a descriptive base filename for output files."""
    stem = Path(csv_path).stem
    return f"{stem}_{tag}" if tag else stem

# -------------------- Core Pipeline --------------------
def run_pipeline(args: argparse.Namespace) -> Bunch:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    
    # Construct a descriptive tag for output filenames based on filters
    name_tag = []
    if args.include_industries:
        name_tag.append("inc-" + "-".join(sorted([s.strip().lower() for s in args.include_industries.split(",") if s.strip()])))
    if args.exclude_industries:
        name_tag.append("exc-" + "-".join(sorted([s.strip().lower() for s in args.exclude_industries.split(",") if s.strip()])))
    if args.start_date or args.end_date:
        name_tag.append(f"{args.start_date or 'min'}_{args.end_date or 'max'}")
    tag = "_".join(name_tag)
    out_name = build_outputs_name(args.csv, tag)

    # --- Data Loading & Filtering ---
    df = pd.read_csv(args.csv)
    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    for c in text_cols:
        if c not in df.columns:
            raise ValueError(f"Missing text column '{c}' in {args.csv}")

    industries_col = args.industries_col if args.industries_col in df.columns else None
    time_col = args.time_col if args.time_col in df.columns else None

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=True)
        df.dropna(subset=[time_col], inplace=True)
        if args.start_date:
            df = df[df[time_col] >= pd.to_datetime(args.start_date, utc=True)]
        if args.end_date:
            df = df[df[time_col] <= pd.to_datetime(args.end_date, utc=True)]

    if industries_col and (args.include_industries or args.exclude_industries):
        inc_set = set([s.strip().lower() for s in (args.include_industries or "").split(",") if s.strip()])
        exc_set = set([s.strip().lower() for s in (args.exclude_industries or "").split(",") if s.strip()])
        inds = df[industries_col].apply(parse_industry_list)
        mask = pd.Series(True, index=df.index)
        if inc_set:
            mask &= inds.apply(lambda xs: in_any_industry(xs, inc_set))
        if exc_set:
            mask &= ~inds.apply(lambda xs: in_any_industry(xs, exc_set))
        df = df.loc[mask].copy()

    df["__text__"] = df.apply(lambda r: clean_join_text(r, text_cols), axis=1)
    if args.min_chars > 0:
        df = df[df["__text__"].str.len() >= args.min_chars]
    if args.min_words > 0:
        df = df[df["__text__"].str.split().apply(len) >= args.min_words]
    df = df.reset_index(drop=True)

    if len(df) < 30:
        raise ValueError(f"Too few rows ({len(df)}) after filtering. Relax filters or widen date/industry.")
    print(f"Loaded {len(df)} rows from {args.csv} after filters.")

    # --- VADER weak labels ---
    sia = ensure_vader_ready()
    if sia is None:
        raise RuntimeError("VADER is not available. Please install nltk and/or vaderSentiment.")
    
    compounds = df["__text__"].apply(lambda t: sia.polarity_scores(str(t))["compound"]).astype(np.float32).values
    y = np.where(compounds <= -0.05, 0, np.where(compounds >= 0.05, 2, 1)).astype(int)
    
    cls, counts = np.unique(y, return_counts=True)
    dist = {int(c): int(n) for c, n in zip(cls, counts)}
    print(f"VADER label distribution: {dist}")

    # --- Split, Vectorize, and Train Model ---
    Xtr_txt, Xte_txt, ytr, yte = train_test_split(df["__text__"], y, test_size=0.2, stratify=y, random_state=args.seed)
    
    vec = TfidfVectorizer(
        ngram_range=(1, args.max_ngram),
        min_df=args.min_df,
        max_df=args.max_df,
        stop_words="english",
        dtype=np.float32
    )
    Xtr = vec.fit_transform(Xtr_txt)
    Xte = vec.transform(Xte_txt)
    
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, multi_class="auto")
    
    # --- Cross-validation and Holdout Evaluation ---
    cv = cross_validate(lr, Xtr, ytr, cv=5, n_jobs=-1, scoring={"acc":"accuracy","f1":"f1_macro"})
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

    # --- Save Metrics and Reports ---
    pd.DataFrame([
        {"metric":"cv_acc_mean", "value":cv_acc_mean},
        {"metric":"cv_acc_std", "value":cv_acc_std},
        {"metric":"cv_f1_mean", "value":cv_f1_mean},
        {"metric":"cv_f1_std", "value":cv_f1_std},
        {"metric":"holdout_acc", "value":acc},
        {"metric":"holdout_macro_f1", "value":f1},
        {"metric":"holdout_macro_auc", "value":auc},
        {"metric":"n_train", "value":int(Xtr.shape[0])},
        {"metric":"n_test", "value":int(Xte.shape[0])}
    ]).to_csv(out_dir / f"lr_metrics_marketaux_{out_name}.csv", index=False)

    with open(out_dir / f"lr_classification_report_marketaux_{out_name}.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(yte, pred, digits=3))

    # ---------- SHAP (REVISED & CORRECTED SECTION) ----------
    try:
        import shap
        print("Computing SHAP values...")
        
        background = shap.sample(Xtr, min(100, Xtr.shape[0]), random_state=args.seed)
        explainer = shap.LinearExplainer(lr, background, feature_names=vec.get_feature_names_out())
        shap_values = explainer.shap_values(Xte)

        # --- REVISED: Summary Plot Generation ---
        plt.figure()
        # The key fix is to pass the list of SHAP arrays directly to summary_plot.
        # The function is designed to handle this by creating a multi-bar plot.
        # We also explicitly pass max_display to ensure the correct number of features are shown.
        shap.summary_plot(
            shap_values, 
            Xte, 
            feature_names=vec.get_feature_names_out(),
            max_display=args.top_k_tokens,
            plot_type="bar",
            class_names=['Negative', 'Neutral', 'Positive'],
            show=False
        )
        plt.tight_layout()
        save_path = out_dir / f"shap_summary_marketaux_{out_name}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Saved revised SHAP summary plot to: {save_path}")

        # --- Token Impact Table Generation ---
        def _to_ndarray_2d(x):
            if isinstance(x, (list, tuple)):
                dense_arrs = [s.toarray() if hasattr(s, 'toarray') else np.asarray(s) for s in x]
                return np.mean(np.stack(dense_arrs, axis=0), axis=0)
            return x.toarray() if hasattr(x, 'toarray') else np.asarray(x)

        sv = _to_ndarray_2d(shap_values)
        names = np.array(vec.get_feature_names_out(), dtype=object)
        pos_mean = np.asarray(np.mean(np.where(sv > 0, sv, 0), axis=0)).ravel()
        neg_mean = np.asarray(np.mean(np.where(sv < 0, sv, 0), axis=0)).ravel()
        
        K = int(args.top_k_tokens)
        pos_idx = np.argsort(-pos_mean)[:K]
        neg_idx = np.argsort(neg_mean)[:K]

        max_len = max(len(pos_idx), len(neg_idx))
        pos_tokens = (names[pos_idx].tolist() + [''] * (max_len - len(pos_idx)))[:max_len]
        pos_vals = (pos_mean[pos_idx].tolist() + [np.nan] * (max_len - len(pos_idx)))[:max_len]
        neg_tokens = (names[neg_idx].tolist() + [''] * (max_len - len(neg_idx)))[:max_len]
        neg_vals = (neg_mean[neg_idx].tolist() + [np.nan] * (max_len - len(neg_idx)))[:max_len]

        token_df = pd.DataFrame({
            "Positive token": pos_tokens,
            "Mean SHAP (+)": pos_vals,
            "Negative token": neg_tokens,
            "Mean SHAP (–)": neg_vals,
        })
        token_df.to_csv(out_dir / f"top_tokens_shap_marketaux_{out_name}.csv", index=False)
        print(f"✅ Saved top tokens table to outputs/")

    except ImportError:
        print("[SHAP] `shap` library not found. Skipping SHAP analysis. Please run `pip install shap`.")
    except Exception as e:
        print(f"[SHAP] Skipped due to an error: {repr(e)}")

    return Bunch(n_rows=len(df), label_dist=dist, out_name=out_name)

# -------------------- CLI --------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Analyze Marketaux news with an interpretable LR+SHAP model.")
    ap.add_argument("--csv", required=True, help="Path to Marketaux articles CSV.")
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
    
    try:
        run_pipeline(args)
    except Exception as e:
        print(f"An error occurred during the pipeline execution: {e}")