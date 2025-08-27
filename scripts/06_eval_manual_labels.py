#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_eval_manual_labels.py
Evaluate VADER, TF-IDF+LogReg, and FinBERT on your manually annotated articles.

Inputs
------
--annot_csv: CSV with columns:
    article_id, published_at, source, url, title, description, industries,
    dominant_industry, manual_sentiment, manual_rationale, manual_scope
  (We will use manual_sentiment as gold label {0=neg, 1=neu, 2=pos}.)
--train_csvs: one or more CSVs (comma-separated) in the familiar "text,y" format
  to train the LR baseline and to tune VADER thresholds (e.g., FPB, FiQA).

Outputs (to ./outputs/)
-----------------------
- 06_preds_{name}.csv
    One row per annotated article with y_true, VADER/LR/FinBERT predictions & probabilities
- 06_metrics_{name}.json
    Summary metrics for each model (Accuracy, Macro-F1, Macro-ROC-AUC where applicable)
- 06_report_{name}_{MODEL}.txt
    Classification report for each model
- 06_confmat_{name}_{MODEL}.csv
    Confusion matrices (labels: 0,1,2)

Notes
-----
- VADER: we grid-search thresholds on the training CSV(s) only, then apply to the gold set.
- LR: vectorizer ngram (1..3), class_weight="balanced". Trained on provided train_csvs.
- FinBERT: ProsusAI/finbert. We automatically map its label strings to {0,1,2}.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

# --- sklearn bits ---
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
)

# --- Optional VADER ---
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    _NLTK_OK = True
except Exception:
    _NLTK_OK = False

# --- FinBERT / Transformers ---
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------- Utils ---------------------------------

def ensure_outputs_dir():
    out = Path("outputs"); out.mkdir(exist_ok=True)
    return out

def combine_text(df, cols):
    """Combine given text columns with '. ' separator, handle NaNs."""
    parts = []
    for c in cols:
        if c not in df.columns:
            df[c] = ""
        parts.append(df[c].fillna("").astype(str))
    txt = parts[0]
    for p in parts[1:]:
        txt = txt + ". " + p
    return txt.str.strip()

def macro_roc_auc_ovr(y_true, proba):
    """Macro AUC (one-vs-rest)."""
    y_true = np.asarray(y_true)
    classes = np.unique(y_true)
    aucs = []
    for k in classes:
        try:
            aucs.append(roc_auc_score((y_true == k).astype(int), proba[:, int(k)]))
        except Exception:
            pass
    if len(aucs) == 0:
        return float("nan")
    return float(np.mean(aucs))

def grid_search_thresholds(scores, y, step=0.01, lo=-0.8, hi=0.8):
    """
    Tune t_neg < t_pos on train scores to maximize macro-F1.
    Return (t_neg, t_pos, best_macro_f1).
    """
    best = (-0.05, 0.05, -1.0)
    grid = np.arange(lo, hi + 1e-9, step)
    for t_neg in grid:
        for t_pos in grid:
            if t_pos <= t_neg:
                continue
            pred = np.where(scores <= t_neg, 0, np.where(scores >= t_pos, 2, 1))
            f1 = f1_score(y, pred, average="macro")
            if f1 > best[2]:
                best = (float(t_neg), float(t_pos), float(f1))
    return best

def safe_int_labels(s):
    """Convert manual_sentiment column to int {0,1,2}, drop others."""
    try:
        v = int(s)
        if v in (0,1,2):
            return v
    except Exception:
        pass
    return None

def detect_device(device_arg: str):
    """Return torch.device and boolean fp16_ok (CUDA available)."""
    if device_arg.lower() == "cpu":
        return torch.device("cpu"), False
    if torch.cuda.is_available():
        return torch.device("cuda"), True
    return torch.device("cpu"), False

# --------------------------------- VADER ---------------------------------

def run_vader_eval(train_texts, train_y, test_texts, step=0.01, lo=-0.8, hi=0.8):
    if not _NLTK_OK:
        return None
    try:
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()
    except Exception:
        return None

    def score_block(texts):
        out = np.zeros(len(texts), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = sia.polarity_scores(str(t))["compound"]
        return out

    tr_scores = score_block(train_texts)
    t_neg, t_pos, _ = grid_search_thresholds(tr_scores, np.asarray(train_y), step=step, lo=lo, hi=hi)

    te_scores = score_block(test_texts)
    te_pred = np.where(te_scores <= t_neg, 0, np.where(te_scores >= t_pos, 2, 1))

    return {
        "pred": te_pred,
        "scores": te_scores,
        "t_neg": t_neg,
        "t_pos": t_pos
    }

# --------------------------------- LR ---------------------------------

def fit_lr(train_texts, train_y):
    vec = TfidfVectorizer(
        ngram_range=(1,3),
        min_df=3,
        max_df=0.9,
        stop_words="english",
        dtype=np.float32
    )
    Xtr = vec.fit_transform(train_texts)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="auto", n_jobs=-1)
    lr.fit(Xtr, np.asarray(train_y))
    return vec, lr

def predict_lr(vec, lr, texts):
    X = vec.transform(texts)
    proba = lr.predict_proba(X)  # (N,3)
    pred = proba.argmax(1)
    return pred, proba

# --------------------------------- FinBERT ---------------------------------

def load_finbert(model_name, device, grad_ckpt=False):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)
    model.eval()
    return tok, model

def finbert_label_map(model):
    """
    Map model id2label (strings) to our ints {0=neg,1=neu,2=pos}.
    We search the label strings for 'neg','neu','pos'.
    """
    id2label = getattr(model.config, "id2label", None)
    if not id2label:
        # sensible default for ProsusAI/finbert
        # commonly: 0=negative, 1=neutral, 2=positive
        return {0:0, 1:1, 2:2}
    mapping = {}
    for i, name in id2label.items():
        s = str(name).lower()
        if "neg" in s:
            mapping[i] = 0
        elif "neu" in s:
            mapping[i] = 1
        elif "pos" in s or "favorable" in s:
            mapping[i] = 2
        else:
            # fallback by index to keep something deterministic
            mapping[i] = i
    return mapping

@torch.no_grad()
def run_finbert(tok, model, texts, device, max_length=128, batch_size=16, fp16=False):
    preds = []
    probs = []
    use_autocast = (fp16 and device.type == "cuda")
    # Use new torch.amp API to silence deprecation warnings
    amp_ctx = torch.amp.autocast("cuda") if use_autocast else torch.no_grad()
    # build mapping
    idmap = finbert_label_map(model)
    # simple batching
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with (torch.amp.autocast("cuda") if use_autocast else torch.no_grad()):
            out = model(**enc)
            logits = out.logits  # (B, num_labels)
            p = torch.softmax(logits, dim=1)  # probs
        # map to our label order
        # Build an array with columns [neg, neu, pos] based on idmap
        num_labels = logits.shape[1]
        p_cpu = p.cpu().numpy()
        # create 3-col prob table initialized with zeros
        p_std = np.zeros((p_cpu.shape[0], 3), dtype=np.float32)
        for src_id in range(num_labels):
            tgt = idmap.get(src_id, src_id)
            if tgt in (0,1,2):
                p_std[:, tgt] = p_cpu[:, src_id]
        probs.append(p_std)
        preds.append(np.argmax(p_std, axis=1))
    probs = np.vstack(probs) if probs else np.zeros((0,3), dtype=np.float32)
    preds = np.concatenate(preds) if preds else np.array([], dtype=int)
    return preds, probs

# --------------------------------- Main ---------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot_csv", required=True, help="Path to annotated_articles.csv")
    ap.add_argument("--train_csvs", required=True, help="Comma-separated CSVs with columns text,y (e.g., fpb.csv[,fiqa_headlines.csv])")
    ap.add_argument("--text_cols", default="title,description", help="Comma-separated text columns to combine from annotation CSV")
    ap.add_argument("--finbert_model", default="ProsusAI/finbert")
    ap.add_argument("--device", default="auto", help="'auto'|'cpu'|'cuda'")
    ap.add_argument("--fp16", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--skip_vader", type=int, default=0)
    args = ap.parse_args()

    out_dir = ensure_outputs_dir()
    name = Path(args.annot_csv).stem

    # ------- Load gold set -------
    df_annot = pd.read_csv(args.annot_csv)
    # label cleaning
    df_annot["__y"] = df_annot["manual_sentiment"].apply(safe_int_labels)
    df_annot = df_annot.dropna(subset=["__y"]).reset_index(drop=True)
    df_annot["__y"] = df_annot["__y"].astype(int)

    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    df_annot["__text"] = combine_text(df_annot, text_cols)
    X_gold = df_annot["__text"].tolist()
    y_gold = df_annot["__y"].tolist()

    if len(df_annot) == 0:
        raise SystemExit("No labeled rows found in annotated CSV (manual_sentiment must be 0/1/2).")

    # ------- Load & combine training CSV(s) for LR + VADER tuning -------
    train_csvs = [p.strip() for p in args.train_csvs.split(",") if p.strip()]
    train_frames = []
    for p in train_csvs:
        d = pd.read_csv(p).dropna(subset=["text","y"])
        d["y"] = d["y"].astype(int)
        train_frames.append(d[["text","y"]])
    if not train_frames:
        raise SystemExit("No training CSVs loaded.")
    train_df = pd.concat(train_frames, ignore_index=True)
    Xtr = train_df["text"].astype(str).tolist()
    ytr = train_df["y"].astype(int).tolist()

    # ------- VADER -------
    vader_res = None
    if not args.skip_vader:
        vader_res = run_vader_eval(Xtr, ytr, X_gold, step=0.01, lo=-0.8, hi=0.8)

    # ------- LR baseline -------
    vec, lr = fit_lr(Xtr, ytr)
    lr_pred, lr_proba = predict_lr(vec, lr, X_gold)

    # ------- FinBERT -------
    device, fp16_ok = detect_device(args.device)
    use_fp16 = (args.fp16 == 1 and fp16_ok)
    tok, fb_model = load_finbert(args.finbert_model, device, grad_ckpt=False)
    fb_pred, fb_proba = run_finbert(
        tok, fb_model, X_gold, device,
        max_length=args.max_length, batch_size=args.batch_size, fp16=use_fp16
    )

    # ------- Metrics -------
    out_metrics = {}

    # LR
    lr_acc = float(accuracy_score(y_gold, lr_pred))
    lr_mf1 = float(f1_score(y_gold, lr_pred, average="macro"))
    lr_auc = float(macro_roc_auc_ovr(y_gold, lr_proba))
    out_metrics["LR_TFIDF"] = {"acc": lr_acc, "macro_f1": lr_mf1, "macro_auc": lr_auc}

    # FinBERT
    fb_acc = float(accuracy_score(y_gold, fb_pred))
    fb_mf1 = float(f1_score(y_gold, fb_pred, average="macro"))
    fb_auc = float(macro_roc_auc_ovr(y_gold, fb_proba))
    out_metrics["FinBERT"] = {"acc": fb_acc, "macro_f1": fb_mf1, "macro_auc": fb_auc}

    # VADER (no AUC; we only have a scalar compound score)
    if vader_res is not None:
        v_pred = vader_res["pred"]
        v_acc = float(accuracy_score(y_gold, v_pred))
        v_mf1 = float(f1_score(y_gold, v_pred, average="macro"))
        out_metrics["VADER"] = {
            "acc": v_acc, "macro_f1": v_mf1,
            "t_neg": float(vader_res["t_neg"]), "t_pos": float(vader_res["t_pos"])
        }

    # ------- Reports & Confusion matrices -------
    # LR
    (out_dir / f"06_report_{name}_LR.txt").write_text(
        classification_report(y_gold, lr_pred, digits=3)
    )
    pd.DataFrame(
        confusion_matrix(y_gold, lr_pred, labels=[0,1,2]),
        index=["true_0","true_1","true_2"],
        columns=["pred_0","pred_1","pred_2"]
    ).to_csv(out_dir / f"06_confmat_{name}_LR.csv", index=True)

    # FinBERT
    (out_dir / f"06_report_{name}_FinBERT.txt").write_text(
        classification_report(y_gold, fb_pred, digits=3)
    )
    pd.DataFrame(
        confusion_matrix(y_gold, fb_pred, labels=[0,1,2]),
        index=["true_0","true_1","true_2"],
        columns=["pred_0","pred_1","pred_2"]
    ).to_csv(out_dir / f"06_confmat_{name}_FinBERT.csv", index=True)

    # VADER
    if vader_res is not None:
        v_pred = vader_res["pred"]
        (out_dir / f"06_report_{name}_VADER.txt").write_text(
            classification_report(y_gold, v_pred, digits=3)
        )
        pd.DataFrame(
            confusion_matrix(y_gold, v_pred, labels=[0,1,2]),
            index=["true_0","true_1","true_2"],
            columns=["pred_0","pred_1","pred_2"]
        ).to_csv(out_dir / f"06_confmat_{name}_VADER.csv", index=True)

    # ------- Per-row predictions table -------
    rows = {
        "article_id": df_annot["article_id"],
        "published_at": df_annot.get("published_at",""),
        "source": df_annot.get("source",""),
        "url": df_annot.get("url",""),
        "title": df_annot.get("title",""),
        "description": df_annot.get("description",""),
        "industries": df_annot.get("industries",""),
        "y_true": df_annot["__y"].values,
        # LR
        "lr_pred": lr_pred,
        "lr_p0": lr_proba[:,0], "lr_p1": lr_proba[:,1], "lr_p2": lr_proba[:,2],
        # FinBERT
        "finbert_pred": fb_pred,
        "finbert_p0": fb_proba[:,0], "finbert_p1": fb_proba[:,1], "finbert_p2": fb_proba[:,2],
    }
    if vader_res is not None:
        rows["vader_pred"] = vader_res["pred"]
        rows["vader_compound"] = vader_res["scores"]

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_dir / f"06_preds_{name}.csv", index=False)

    # ------- Save metrics summary -------
    (out_dir / f"06_metrics_{name}.json").write_text(json.dumps(out_metrics, indent=2))

    # ------- Print summary -------
    print("\n=== Results on manual gold ===")
    for m, v in out_metrics.items():
        acc = v.get("acc")
        mf1 = v.get("macro_f1")
        auc = v.get("macro_auc", float("nan"))
        if np.isnan(auc):
            print(f"{m:10s}  Acc={acc:.3f}  Macro-F1={mf1:.3f}")
        else:
            print(f"{m:10s}  Acc={acc:.3f}  Macro-F1={mf1:.3f}  Macro-AUC={auc:.3f}")
    print(f"\nSaved: {out_dir / f'06_preds_{name}.csv'}")
    print(f"Saved: {out_dir / f'06_metrics_{name}.json'}")
    print("Plus per-model reports & confusion matrices.")
    
if __name__ == "__main__":
    main()
