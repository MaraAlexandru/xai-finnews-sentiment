#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_lexicon_ablation.py
----------------------

Goal
====
Benchmark a domain-specific lexicon (Loughran–McDonald) against your existing baselines.
Two ablations:
  (A) Pure lexicon rule-based sentiment via LM counts + tuned thresholds
  (B) A simple classifier that uses ONLY LM-derived features (counts/ratios), i.e., no TF–IDF

Why
===
- Directly addresses reviewers' requests to replace VADER with a finance-specific lexicon.
- Shows whether finance-aware priors help, and by how much, on your benchmark datasets.
- Produces paper-ready artifacts: metrics JSON, confusion matrices, and per-example tables.

Inputs
======
CSV with columns:
  text : string
  y    : {0=negative, 1=neutral, 2=positive}

Lexicon files (auto-download if missing; or place manually under data/lexicons/lm/):
  positive.txt, negative.txt, uncertainty.txt, litigious.txt,
  modal_weak.txt, modal_strong.txt, constraining.txt

Outputs (to ./outputs/)
=======================
- lm_scored_{name}.csv             # per-row counts/hits + rule-based scores and predictions
- lm_threshold_metrics_{name}.json # tuned-threshold results (Acc/Macro-F1) + chosen thresholds
- lm_threshold_confmat_{name}.csv  # confusion matrix for rule-based method
- lm_lr_counts_metrics_{name}.json # LR-on-LM-features results (CV + holdout)
- lm_lr_counts_confmat_{name}.csv  # confusion matrix for LM-features classifier
- lm_lr_counts_coefs_{name}.csv    # interpretable coefficients of the LM-features LR

Optional (if --compare_vader 1 and nltk is installed):
- lm_vs_vader_{name}.csv           # head-to-head scores (for appendix/plots)
- vader_threshold_metrics_{name}.json, vader_threshold_confmat_{name}.csv

Usage examples
==============
python scripts/04_lexicon_ablation.py --csv data/processed/fpb.csv
python scripts/04_lexicon_ablation.py --csv data/processed/fiqa_headlines.csv --compare_vader 1

Tips
====
- On tiny/imbalanced sets (FiQA), add --min_count 1 (count words that appear once).
- If you want faster runs on big CSVs, reduce --grid_step from 0.01 to 0.02.
"""

import argparse, json, re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Optional: VADER comparison
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    _NLTK_OK = True
except Exception:
    _NLTK_OK = False


# ----------------------------- 1) Loughran–McDonald loader -----------------------------
# We try to be fully reproducible: first look for local files, else attempt lightweight download
DEFAULT_LM_DIR = Path("data/lexicons/lm")

LM_FILES = {
    "positive":      "positive.txt",
    "negative":      "negative.txt",
    "uncertainty":   "uncertainty.txt",
    "litigious":     "litigious.txt",
    "modal_weak":    "modal_weak.txt",
    "modal_strong":  "modal_strong.txt",
    "constraining":  "constraining.txt",
}

# A couple of public mirrors (kept generic; if offline, just place files locally and it will use them)
LM_URLS = {
    "positive":     "https://raw.githubusercontent.com/kwcooper1/loughran-mcdonald/master/LM_PositiveWords.txt",
    "negative":     "https://raw.githubusercontent.com/kwcooper1/loughran-mcdonald/master/LM_NegativeWords.txt",
    "uncertainty":  "https://raw.githubusercontent.com/kwcooper1/loughran-mcdonald/master/LM_Uncertainty.txt",
    "litigious":    "https://raw.githubusercontent.com/kwcooper1/loughran-mcdonald/master/LM_Litigious.txt",
    "modal_weak":   "https://raw.githubusercontent.com/kwcooper1/loughran-mcdonald/master/LM_WeakModal.txt",
    "modal_strong": "https://raw.githubusercontent.com/kwcooper1/loughran-mcdonald/master/LM_StrongModal.txt",
    "constraining": "https://raw.githubusercontent.com/kwcooper1/loughran-mcdonald/master/LM_Constraining.txt",
}

def _download(url: str, out_path: Path):
    try:
        import urllib.request
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=30) as r, open(out_path, "wb") as f:
            f.write(r.read())
        return True
    except Exception:
        return False

def load_lm_lexicons(lex_dir: Path = DEFAULT_LM_DIR):
    """
    Returns: dict[str -> set[str]] mapping category name -> set of lowercase words
    If files are missing, tries to download from mirrors. If that fails, raises.
    """
    lex_dir = Path(lex_dir)
    lex_dir.mkdir(parents=True, exist_ok=True)
    lex = {}
    for cat, fname in LM_FILES.items():
        fp = lex_dir / fname
        if not fp.exists():
            # try a mirror download
            if cat in LM_URLS:
                ok = _download(LM_URLS[cat], fp)
                if not ok:
                    raise FileNotFoundError(f"Missing {fp} and download failed; please place the LM file manually.")
            else:
                raise FileNotFoundError(f"Missing lexicon file {fp}.")
        # read list
        words = []
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                w = line.strip().lower()
                if not w or w.startswith("#"):  # skip comments if any
                    continue
                # LM lists sometimes contain markers or tabs; keep first token
                w = w.split()[0]
                words.append(w)
        lex[cat] = set(words)
    return lex


# ----------------------------- 2) Tokenization & counting -----------------------------
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-\']+")  # words starting with a letter

NEGATORS = {
    "no","not","never","none","nobody","nothing","neither","nowhere","hardly","scarcely","barely",
    "n't","cannot","without","less","few","little"
}

def simple_tokenize(text: str):
    # Lowercase, keep alphanum words with hyphens/apostrophes (LM lists often plain words)
    return [m.group(0).lower() for m in TOKEN_RE.finditer(str(text))]

def count_with_negation(tokens, lm_lex, neg_window=3, min_count=2):
    """
    Count LM categories with a small negation scope for pos/neg ONLY.
    - neg_window: how many previous tokens act as negators
    - min_count: ignore tokens shorter than min_count chars (e.g., 'a', 'i') to reduce noise

    Returns: dict of counts and also the specific hit lists for pos/neg categories (for the CSV)
    """
    pos_hits, neg_hits = [], []
    counts = {cat: 0 for cat in lm_lex.keys()}
    N = len(tokens)
    for i, w in enumerate(tokens):
        if len(w) < min_count:
            continue

        # Check all LM categories
        hit_any = False
        for cat, vocab in lm_lex.items():
            if w in vocab:
                counts[cat] += 1
                hit_any = True

        # Negation only flips sentiment categories
        if w in lm_lex["positive"] or w in lm_lex["negative"]:
            # Look back a short window for negators
            scope = tokens[max(0, i-neg_window):i]
            is_negated = any(n in NEGATORS for n in scope)
            if w in lm_lex["positive"]:
                pos_hits.append(("~"+w if is_negated else w))
            if w in lm_lex["negative"]:
                neg_hits.append(("~"+w if is_negated else w))

            if is_negated:
                # Instead of counting both, we flip: decrement original, increment opposite
                if w in lm_lex["positive"]:
                    counts["positive"] -= 1
                    counts["negative"] += 1
                elif w in lm_lex["negative"]:
                    counts["negative"] -= 1
                    counts["positive"] += 1

    # basic guards (no negatives below zero)
    counts["positive"] = max(0, counts["positive"])
    counts["negative"] = max(0, counts["negative"])
    return counts, pos_hits, neg_hits


# ----------------------------- 3) Rule-based label via tuned thresholds -----------------------------
def polarity_score(pos_count, neg_count, smooth=1e-6):
    # classic normalized difference
    return (pos_count - neg_count) / max(smooth, (pos_count + neg_count))

def grid_search_thresholds(polarity_train, y_train, step=0.01, lo=-0.5, hi=0.5):
    """
    Tune two thresholds t_neg < t_pos:
      label = -1 if s <= t_neg; 0 if t_neg < s < t_pos; +1 if s >= t_pos
    Choose t_neg, t_pos that maximize macro-F1 on train set.
    Returns (t_neg, t_pos, best_macro_f1)
    """
    best = (-0.05, 0.05, -1.0)  # default near zero
    grid = np.arange(lo, hi + 1e-9, step)
    for t_neg in grid:
        for t_pos in grid:
            if t_pos <= t_neg:
                continue
            pred = np.where(polarity_train <= t_neg, 0,
                    np.where(polarity_train >= t_pos, 2, 1))
            f1 = f1_score(y_train, pred, average="macro")
            if f1 > best[2]:
                best = (float(t_neg), float(t_pos), float(f1))
    return best


# ----------------------------- 4) LR on LM features (no TF–IDF) -----------------------------
def build_lm_feature_matrix(df_counts: pd.DataFrame):
    """
    From the per-row counts table, build an interpretable feature set for LR:
      - raw counts: pos, neg, uncertainty, litigious, modal_weak, modal_strong, constraining
      - total tokens, pos_ratio, neg_ratio, uncertainty_ratio
      - (pos - neg), (pos + neg)
    Returns X (numpy), feature_names (list)
    """
    cols = ["positive","negative","uncertainty","litigious","modal_weak","modal_strong","constraining","n_tokens"]
    for c in cols:
        if c not in df_counts.columns:
            df_counts[c] = 0

    X_list = []
    names = []

    def add(colname, arr):
        nonlocal X_list, names
        X_list.append(arr.reshape(-1,1))
        names.append(colname)

    pos = df_counts["positive"].values.astype(np.float32)
    neg = df_counts["negative"].values.astype(np.float32)
    tot = df_counts["n_tokens"].values.astype(np.float32) + 1e-6

    add("pos", pos)
    add("neg", neg)
    add("unc", df_counts["uncertainty"].values.astype(np.float32))
    add("lit", df_counts["litigious"].values.astype(np.float32))
    add("mweak", df_counts["modal_weak"].values.astype(np.float32))
    add("mstrong", df_counts["modal_strong"].values.astype(np.float32))
    add("constr", df_counts["constraining"].values.astype(np.float32))

    add("tokens", tot)
    add("pos_ratio", pos/tot)
    add("neg_ratio", neg/tot)
    add("unc_ratio", df_counts["uncertainty"].values.astype(np.float32)/tot)
    add("pos_minus_neg", pos - neg)
    add("pos_plus_neg", pos + neg)

    X = np.hstack(X_list)
    return X, names


# ----------------------------- 5) Main pipeline -----------------------------
def main(args):
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)
    name = Path(args.csv).stem

    # Load data
    df = pd.read_csv(args.csv).dropna(subset=["text","y"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df["y"]    = df["y"].astype(int)
    print(f"Loaded {len(df)} rows from {args.csv}")

    # Split for tuning & holdout
    Xtr_txt, Xte_txt, ytr, yte = train_test_split(
        df["text"], df["y"], test_size=0.2, stratify=df["y"], random_state=args.seed
    )

    # Load LM lexicons
    lm_lex = load_lm_lexicons(Path(args.lexicon_dir))

    # --- Count features per row (train & test) ---
    def score_block(texts):
        rows = []
        for t in texts:
            toks = simple_tokenize(t)
            counts, pos_hits, neg_hits = count_with_negation(
                toks, lm_lex, neg_window=args.neg_window, min_count=args.min_count
            )
            row = {
                "n_tokens": len(toks),
                **{k:int(v) for k,v in counts.items()},
                "pos_hits": ";".join(pos_hits[:50]),
                "neg_hits": ";".join(neg_hits[:50]),
            }
            row["polarity"] = polarity_score(row["positive"], row["negative"])
            rows.append(row)
        return pd.DataFrame(rows)

    tr_counts = score_block(Xtr_txt)
    te_counts = score_block(Xte_txt)

    # --- Rule-based (tuned thresholds) ---
    t_neg, t_pos, best_f1 = grid_search_thresholds(
        tr_counts["polarity"].values, ytr.values,
        step=args.grid_step, lo=args.grid_lo, hi=args.grid_hi
    )
    # predict on test
    pol = te_counts["polarity"].values
    y_pred_thr = np.where(pol <= t_neg, 0, np.where(pol >= t_pos, 2, 1))

    thr_acc = accuracy_score(yte, y_pred_thr)
    thr_mf1 = f1_score(yte, y_pred_thr, average="macro")
    print(f"[LM thresholds] Acc={thr_acc:.3f}  Macro-F1={thr_mf1:.3f}  (t_neg={t_neg:.2f}, t_pos={t_pos:.2f})")

    # Save rule-based artifacts
    pd.DataFrame({
        "text": Xte_txt.values,
        "y_true": yte.values,
        "polarity": pol,
        "y_pred": y_pred_thr
    }).to_csv(out_dir / f"lm_rule_based_preds_{name}.csv", index=False)

    pd.DataFrame(
        confusion_matrix(yte, y_pred_thr, labels=[0,1,2]),
        index=["true_neg","true_neu","true_pos"],
        columns=["pred_neg","pred_neu","pred_pos"]
    ).to_csv(out_dir / f"lm_threshold_confmat_{name}.csv")

    (out_dir / f"lm_threshold_metrics_{name}.json").write_text(json.dumps({
        "dataset": name,
        "tuned_on": "train split (80%)",
        "t_neg": t_neg, "t_pos": t_pos, "train_best_macro_f1": best_f1,
        "holdout_accuracy": float(thr_acc),
        "holdout_macro_f1": float(thr_mf1),
        "n_test": int(len(Xte_txt))
    }, indent=2))

    # --- Build per-row table for appendix (counts + hits + final rule label) ---
    te_table = te_counts.copy()
    te_table.insert(0, "text", Xte_txt.values)
    te_table.insert(1, "y", yte.values)
    te_table["rule_pred"] = y_pred_thr
    te_table.to_csv(out_dir / f"lm_scored_{name}.csv", index=False)

    # --- LR classifier on LM-only features (interpretable) ---
    X_tr, feat_names = build_lm_feature_matrix(tr_counts)
    X_te, _          = build_lm_feature_matrix(te_counts)

    lm_lr = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(
            max_iter=1000, class_weight="balanced", multi_class="auto", n_jobs=-1
        )),
    ])
    cv = cross_validate(lm_lr, X_tr, ytr.values, cv=5,
                        scoring={"acc":"accuracy","f1":"f1_macro"})
    lm_lr.fit(X_tr, ytr.values)
    y_pred_lr = lm_lr.predict(X_te)

    lr_acc = accuracy_score(yte.values, y_pred_lr)
    lr_mf1 = f1_score(yte.values, y_pred_lr, average="macro")
    print(f"[LM LR-features] Acc={lr_acc:.3f}  Macro-F1={lr_mf1:.3f}  (CV acc={np.mean(cv['test_acc']):.3f}±{np.std(cv['test_acc']):.3f})")

    # Save LR-on-LM-feature artifacts
    pd.DataFrame(
        confusion_matrix(yte.values, y_pred_lr, labels=[0,1,2]),
        index=["true_neg","true_neu","true_pos"],
        columns=["pred_neg","pred_neu","pred_pos"]
    ).to_csv(out_dir / f"lm_lr_counts_confmat_{name}.csv")

    (out_dir / f"lm_lr_counts_metrics_{name}.json").write_text(json.dumps({
        "dataset": name,
        "cv_acc_mean": float(np.mean(cv["test_acc"])),
        "cv_acc_std":  float(np.std(cv["test_acc"])),
        "cv_f1_mean":  float(np.mean(cv["test_f1"])),
        "cv_f1_std":   float(np.std(cv["test_f1"])),
        "holdout_accuracy": float(lr_acc),
        "holdout_macro_f1": float(lr_mf1),
        "n_test": int(len(Xte_txt))
    }, indent=2))

    # Extract coefficients for interpretability (log-odds contributions)
    # Note: pipeline -> access final LR and its coef_
    lr_final = lm_lr.named_steps["lr"]
    coefs = pd.DataFrame(lr_final.coef_, columns=feat_names)
    coefs.insert(0, "class", ["neg","neu","pos"])
    coefs.to_csv(out_dir / f"lm_lr_counts_coefs_{name}.csv", index=False)

    # --- Optional: compare to VADER rule-based thresholds on the same split ---
    if args.compare_vader:
        if not _NLTK_OK:
            print("[VADER] nltk not available; skipping.")
        else:
            try:
                nltk.download("vader_lexicon", quiet=True)
                vader = SentimentIntensityAnalyzer()

                def vader_score_block(texts):
                    comp = []
                    for t in texts:
                        comp.append(vader.polarity_scores(str(t))["compound"])
                    return np.array(comp, dtype=np.float32)

                v_tr = vader_score_block(Xtr_txt)
                v_te = vader_score_block(Xte_txt)

                # tune thresholds on train (same grid search), but now scores are in [-1,1]
                v_tneg, v_tpos, v_best = grid_search_thresholds(
                    v_tr, ytr.values, step=args.grid_step, lo=-0.8, hi=0.8
                )
                v_pred = np.where(v_te <= v_tneg, 0, np.where(v_te >= v_tpos, 2, 1))
                v_acc  = accuracy_score(yte.values, v_pred)
                v_mf1  = f1_score(yte.values, v_pred, average="macro")
                print(f"[VADER thresholds] Acc={v_acc:.3f}  Macro-F1={v_mf1:.3f}  (t_neg={v_tneg:.2f}, t_pos={v_tpos:.2f})")

                (out_dir / f"vader_threshold_metrics_{name}.json").write_text(json.dumps({
                    "dataset": name,
                    "t_neg": v_tneg, "t_pos": v_tpos, "train_best_macro_f1": v_best,
                    "holdout_accuracy": float(v_acc),
                    "holdout_macro_f1": float(v_mf1),
                    "n_test": int(len(Xte_txt))
                }, indent=2))

                pd.DataFrame(
                    confusion_matrix(yte.values, v_pred, labels=[0,1,2]),
                    index=["true_neg","true_neu","true_pos"],
                    columns=["pred_neg","pred_neu","pred_pos"]
                ).to_csv(out_dir / f"vader_threshold_confmat_{name}.csv")

                # pairwise table for appendix
                pd.DataFrame({
                    "text": Xte_txt.values,
                    "y_true": yte.values,
                    "lm_polarity": pol,
                    "lm_pred": y_pred_thr,
                    "vader_compound": v_te,
                    "vader_pred": v_pred
                }).to_csv(out_dir / f"lm_vs_vader_{name}.csv", index=False)

            except Exception as e:
                print(f"[VADER] failed with {repr(e)}; skipping.")

    print("Saved all artifacts in outputs/")
    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV with columns text,y")
    ap.add_argument("--lexicon_dir", default=str(DEFAULT_LM_DIR), help="Dir with LM lexicon txt files")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg_window", type=int, default=3, help="Negation scope (tokens back)")
    ap.add_argument("--min_count", type=int, default=2, help="Ignore tokens shorter than this")
    ap.add_argument("--grid_step", type=float, default=0.01)
    ap.add_argument("--grid_lo", type=float, default=-0.5)
    ap.add_argument("--grid_hi", type=float, default=0.5)
    ap.add_argument("--compare_vader", type=int, default=0)
    args = ap.parse_args()
    main(args)
