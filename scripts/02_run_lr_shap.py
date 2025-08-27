# scripts/02_run_lr_shap.py
"""
Train & evaluate TF-IDF + Logistic Regression (transparent baseline) on any CSV with columns text,y.

What it does (and why):
- Vectorize text with TF-IDF ngrams (1..3) → balances specificity/coverage and keeps things interpretable.
- Train multinomial Logistic Regression with class_weight='balanced' → handles class imbalance.
- Report 5-fold CV on the training split (accuracy, macro-F1) → assesses stability.
- Report 80/20 stratified holdout (accuracy, macro-F1, macro-ROC-AUC OvR) → generalization snapshot.
- Compute SHAP values with LinearExplainer (exact for linear models) → global interpretability.
- Save a token impact table (mean positive/negative SHAP per feature) → compact top-driver list.

Inputs:
  --csv data/processed/fpb.csv  (or fiqa_headlines.csv or any dataset with columns: text,y)

Outputs (to outputs/):
  - lr_metrics_{name}.csv                 # CV means/stds & holdout metrics (one row per metric)
  - lr_classification_report_{name}.txt   # per-class precision/recall/F1 for holdout
  - shap_summary_{name}.png               # global SHAP beeswarm (top features by impact)
  - top_tokens_shap_{name}.csv            # 2x20 table: strongest positive & negative mean SHAP
  - lr_results_{name}.json                # small machine-readable summary for your tables
"""

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)


# ---------- small helpers ----------

def macro_roc_auc_ovr(y_true, proba):
    """
    Multi-class AUC via One-vs-Rest macro average:
      For each class k, treat it as positive vs others; average AUCs.
    """
    classes = np.unique(y_true)
    aucs = []
    for k in classes:
        k = int(k)
        if proba.shape[1] <= k:
            continue
        try:
            aucs.append(roc_auc_score((y_true == k).astype(int), proba[:, k]))
        except Exception:
            # If a class is missing in the holdout slice, skip it
            pass
    return float(np.mean(aucs)) if aucs else float("nan")


def to_float(x):
    """Convert numpy types / 0-D arrays to plain Python float safely."""
    try:
        return float(np.asarray(x).item())
    except Exception:
        return float(x)


def shap_to_dense_2d(shap_vals):
    """
    Normalize SHAP return types into a dense 2-D numpy array of shape (n_samples, n_features).

    SHAP can return:
      - ndarray with shape (N, F)
      - scipy sparse matrix (has .toarray())
      - a list/tuple of per-class arrays [C x (N, F)] for multiclass
    We convert all cases to a single (N, F) dense array by:
      - sparse -> .toarray()
      - list-of-arrays -> mean over classes (sign-preserving averaging of raw SHAP values)
    """
    # sparse?
    if hasattr(shap_vals, "toarray"):
        x = shap_vals.toarray()
    elif isinstance(shap_vals, (list, tuple)):
        arrs = [sv.toarray() if hasattr(sv, "toarray") else np.asarray(sv) for sv in shap_vals]
        # Ensure all shapes align; if not, pad/truncate per-class arrays to the min feature count
        minF = min(a.shape[1] for a in arrs)
        arrs = [a[:, :minF] for a in arrs]
        x = np.mean(np.stack(arrs, axis=0), axis=0)  # (N, F)
    else:
        x = np.asarray(shap_vals)

    # If it’s a matrix type, force plain ndarray
    if isinstance(x, np.matrix):
        x = np.asarray(x)

    # Final sanity: ensure 2-D
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x  # (N, F)


def align_feature_names(vec, n_features_needed):
    """
    Get feature names from the fitted vectorizer and align their length to `n_features_needed`.

    Why needed:
    - On some SHAP/scikit versions/platforms, the SHAP matrix may slightly differ in F
      (e.g., due to internal handling). We make the names array match the SHAP width
      by truncating or padding with synthetic placeholders.

    Returns: np.ndarray of length n_features_needed (dtype=object)
    """
    names_full = np.array(vec.get_feature_names_out(), dtype=object)
    F_names = len(names_full)
    F_sv = int(n_features_needed)

    if F_sv == F_names:
        return names_full

    if F_sv < F_names:
        # SHAP returned fewer columns than the vectorizer vocabulary → truncate names
        return names_full[:F_sv]

    # F_sv > F_names: pad with placeholders
    pad = np.array([f"feat_{i}" for i in range(F_names, F_sv)], dtype=object)
    return np.concatenate([names_full, pad], axis=0)


def topk_indices_safely(arr, k, reverse=False, limit=None):
    """
    Return top-k indices of `arr`, guarding against out-of-bounds and NaNs.
      - reverse=False → largest first (descending)
      - reverse=True  → smallest first (ascending)
    `limit` (if provided) clips any indices to be < limit.
    """
    arr = np.asarray(arr)
    # Replace NaNs with extreme values so they end up at the end
    arr = np.where(np.isnan(arr), -np.inf if not reverse else np.inf, arr)
    order = np.argsort(arr)  # ascending
    if not reverse:
        order = order[::-1]   # descending for "largest first"

    if limit is not None:
        order = order[order < int(limit)]

    k = int(min(k, len(order)))
    return order[:k]


# ---------- main pipeline ----------

def main(csv_path: str):
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    name = Path(csv_path).stem

    # 1) Load & shuffle for reproducibility
    df = pd.read_csv(csv_path).dropna(subset=["text", "y"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # 2) Train/holdout split (stratified)
    Xtr_txt, Xte_txt, ytr, yte = train_test_split(
        df["text"], df["y"], test_size=0.2, stratify=df["y"], random_state=42
    )

    # 3) Vectorize text with TF-IDF ngrams (1..3)
    vec = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.9,
        stop_words="english",
        dtype=np.float32
    )
    Xtr = vec.fit_transform(Xtr_txt)
    Xte = vec.transform(Xte_txt)

    # 4) Model: multinomial Logistic Regression with class balancing
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)

    # 5) 5-fold CV on training features (sanity + stability)
    cv = cross_validate(
        lr, Xtr, ytr,
        cv=5, n_jobs=-1,
        scoring={"acc": "accuracy", "f1": "f1_macro"}
    )
    cv_acc_mean = to_float(np.mean(cv["test_acc"]))
    cv_acc_std  = to_float(np.std(cv["test_acc"]))
    cv_f1_mean  = to_float(np.mean(cv["test_f1"]))
    cv_f1_std   = to_float(np.std(cv["test_f1"]))
    print(f"CV: acc={cv_acc_mean:.3f}±{cv_acc_std:.3f}  f1={cv_f1_mean:.3f}±{cv_f1_std:.3f}")

    # 6) Fit on full training split and evaluate holdout
    lr.fit(Xtr, ytr)
    proba = lr.predict_proba(Xte)  # (N, C)
    pred = proba.argmax(1)

    acc = to_float(accuracy_score(yte, pred))
    f1  = to_float(f1_score(yte, pred, average="macro"))
    auc = to_float(macro_roc_auc_ovr(yte, proba))
    print(f"Holdout: acc={acc:.3f}  macro-f1={f1:.3f}  macro-auc={auc:.3f}")

    # 7) Save numeric results (row-wise to avoid dtype pitfalls)
    metrics_rows = [
        {"metric": "cv_acc_mean",       "value": cv_acc_mean},
        {"metric": "cv_acc_std",        "value": cv_acc_std},
        {"metric": "cv_f1_mean",        "value": cv_f1_mean},
        {"metric": "cv_f1_std",         "value": cv_f1_std},
        {"metric": "holdout_acc",       "value": acc},
        {"metric": "holdout_macro_f1",  "value": f1},
        {"metric": "holdout_macro_auc", "value": auc},
    ]
    pd.DataFrame(metrics_rows).to_csv(out_dir / f"lr_metrics_{name}.csv", index=False)

    with open(out_dir / f"lr_classification_report_{name}.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(yte, pred, digits=3))

    # 8) SHAP explanations (global summary + token table)
    print("Computing SHAP values (this can take a moment)...")
    background = shap.sample(Xtr, 100, random_state=0)  # small background set for speed
    explainer  = shap.LinearExplainer(
        lr, background, feature_names=vec.get_feature_names_out()
    )
    shap_vals  = explainer.shap_values(Xte)             # can be array, sparse, or list per class

    # 8a) Global summary plot
    plt.figure()
    shap.summary_plot(shap_vals, Xte, feature_names=vec.get_feature_names_out(), show=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_summary_{name}.png", dpi=250)
    plt.close()

    # 8b) Token impact table
    # Normalize SHAP values to dense (N, F)
    sv = shap_to_dense_2d(shap_vals)       # shape (n_samples, n_features) guaranteed
    n_features_sv = sv.shape[1]

    # Align feature names to this exact width to prevent index mismatches
    names = align_feature_names(vec, n_features_sv)     # len(names) == n_features_sv

    # Compute mean positive / negative contributions per feature (1-D arrays length F)
    pos_mean = np.mean(np.where(sv > 0, sv, 0), axis=0).astype(float)
    neg_mean = np.mean(np.where(sv < 0, sv, 0), axis=0).astype(float)

    # Choose top-k safely, clipping to feature-name length just in case
    TOP_K = 20
    pos_idx = topk_indices_safely(pos_mean, TOP_K, reverse=False, limit=len(names))
    neg_idx = topk_indices_safely(neg_mean, TOP_K, reverse=True,  limit=len(names))  # most negative first

    # Build balanced columns (pad with blanks/NaNs so pandas is happy even if counts differ)
    k_pos, k_neg = len(pos_idx), len(neg_idx)
    rows = max(k_pos, k_neg)

    pos_tokens = names[pos_idx].tolist() + [""] * (rows - k_pos)
    pos_vals   = pos_mean[pos_idx].tolist() + [np.nan] * (rows - k_pos)
    neg_tokens = names[neg_idx].tolist() + [""] * (rows - k_neg)
    neg_vals   = neg_mean[neg_idx].tolist() + [np.nan] * (rows - k_neg)

    token_df = pd.DataFrame({
        "Positive token": pos_tokens,
        "Mean SHAP (+)": pos_vals,
        "Negative token": neg_tokens,
        "Mean SHAP (–)": neg_vals,
    })
    token_df.to_csv(out_dir / f"top_tokens_shap_{name}.csv", index=False)

    # 9) Small JSON summary for easy table building
    with open(out_dir / f"lr_results_{name}.json", "w") as f:
        json.dump({
            "dataset": name,
            "cv_acc_mean":       cv_acc_mean,
            "cv_acc_std":        cv_acc_std,
            "cv_f1_mean":        cv_f1_mean,
            "cv_f1_std":         cv_f1_std,
            "holdout_acc":       acc,
            "holdout_macro_f1":  f1,
            "holdout_macro_auc": auc
        }, f, indent=2)

    print("Saved metrics & plots in outputs/")

# ---------- CLI ----------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV with columns text,y")
    args = ap.parse_args()
    main(args.csv)
