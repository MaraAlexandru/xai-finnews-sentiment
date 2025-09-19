# scripts/16_deep_explain_finbert.py
"""
Deep XAI for FinBERT:
 - Integrated Gradients (Captum) over embeddings
 - LIME (text) local explanations
 - Attention Rollout token importances
 - Faithfulness curves (deleting top-k tokens)

Inputs (priority for samples):
  data/annotation/annotated_articles.csv  (if present, sampled from here; will merge text by article_id)
  else: data/processed/marketaux/marketaux_news_articles.csv  (EN only)

Outputs:
  outputs/deep_xai/examples.csv                 # texts + predictions
  outputs/deep_xai/ig/attributions.jsonl        # per-example token attributions
  outputs/deep_xai/lime/attributions.jsonl
  outputs/deep_xai/attn/attributions.jsonl
  outputs/deep_xai/faithfulness_curves.csv      # method, frac_removed, prob
  outputs/deep_xai/example_cards.md             # human-friendly snippets

Run: python scripts/16_deep_explain_finbert.py
"""
from pathlib import Path
import json, re, random
import numpy as np
import pandas as pd

from _config import (
    MARKETAUX_ARTICLES_CSV, ANNOTATED_GOLD_CSV, OUTPUTS_DIR, SEED
)

random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = OUTPUTS_DIR / "deep_xai"
OUT_DIR_IG = OUT_DIR / "ig"
OUT_DIR_LIME = OUT_DIR / "lime"
OUT_DIR_ATTN = OUT_DIR / "attn"

N_EXAMPLES = 200   # cap to keep runtime reasonable (lower if on CPU only)
MAX_TOKENS = 160   # truncate long texts for visualization (WordPiece limit fed to model)

MODEL_NAME = "ProsusAI/finbert"  # we’ll use config.id2label dynamically

# ---------------------------
# Data loading
# ---------------------------
def _load_samples():
    """Prefer gold annotations; if they lack text, merge with Marketaux by article_id."""
    gold_p = Path(ANNOTATED_GOLD_CSV)
    mkt_p = Path(MARKETAUX_ARTICLES_CSV)

    if gold_p.exists():
        df = pd.read_csv(gold_p)
        # Try to use text/title/description if present
        text_cols = [c for c in ["text", "title", "description"] if c in df.columns]
        if not text_cols and "article_id" in df.columns and mkt_p.exists():
            m = pd.read_csv(mkt_p)[["article_id", "text", "title", "description", "language"]]
            df = df.merge(m, on="article_id", how="left")
            text_cols = [c for c in ["text", "title", "description"] if c in df.columns]

        if text_cols:
            df["text"] = (
                df[text_cols]
                .astype(str)
                .apply(lambda r: " — ".join([z for z in r.values.tolist() if z and z.lower() != "nan"]), axis=1)
            )
        else:
            print("[WARN] Gold file has no text/title/description, and no Marketaux merge available. Falling back to Marketaux.")
            df = None  # fall through to Marketaux

        if df is not None:
            # Prefer EN if language present
            if "language" in df.columns:
                df["language"] = df["language"].fillna("unk").str.lower()
                df = df[df["language"].eq("en") | df["language"].eq("unk")].copy()

    if (gold_p.exists() and df is not None and not df.empty):
        pass
    else:
        if not mkt_p.exists():
            print("[ERROR] No samples available (neither gold with text nor marketaux file exists).")
            return None
        df = pd.read_csv(mkt_p)
        df["language"] = df["language"].fillna("unk").str.lower()
        df = df[df["language"].eq("en")].copy()
        df["text"] = df["text"].fillna("").astype(str)

    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.len() >= 15].copy()
    if df.empty:
        print("[ERROR] No eligible rows after filtering.")
        return None

    if len(df) > N_EXAMPLES:
        df = df.sample(N_EXAMPLES, random_state=SEED).reset_index(drop=True)

    # Trim extremely long strings to avoid super-long runs
    df["text"] = df["text"].str.slice(0, 2000)

    return df[["text"]].reset_index(drop=True)

# ---------------------------
# Model + prediction
# ---------------------------
def _prep_hf():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()
        id2label = getattr(model.config, "id2label", {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
        return tok, model, id2label
    except Exception as e:
        print(f"[ERROR] transformers model load failed: {e}")
        return None, None, None

def _predict_proba(texts, tok, model, batch_size=16):
    """Return numpy array [N, C] with class probs."""
    from transformers import TextClassificationPipeline
    pipe = TextClassificationPipeline(model=model, tokenizer=tok, return_all_scores=True, truncation=True)
    arrs = []
    # manual batching to be nicer on memory/CPU
    for i in range(0, len(texts), batch_size):
        out = pipe(texts[i:i+batch_size])
        for lst in out:
            arrs.append([d["score"] for d in lst])
    return np.array(arrs)

# ---------------------------
# Explanations
# ---------------------------
def _integrated_gradients(texts, tok, model, id2label):
    """Integrated Gradients over input embeddings (Captum)."""
    try:
        from captum.attr import IntegratedGradients
        import torch
    except Exception as e:
        print(f"[WARN] Captum not available: {e}")
        return []

    device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
    model.to(device)

    def forward_emb(inputs_embeds, attention_mask):
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs.logits

    ig = IntegratedGradients(forward_emb)
    results = []

    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        with __import__("torch").no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            probs = __import__("torch").softmax(logits, dim=-1).detach().cpu().numpy()[0]
            pred_cls = int(__import__("torch").argmax(logits, dim=-1).item())

            emb_layer = model.get_input_embeddings()
            emb = emb_layer(input_ids)  # (1, T, H)

        baseline = __import__("torch").zeros_like(emb)

        # Fewer steps for speed; adjust if you want smoother IG
        attributions, _ = ig.attribute(
            inputs=emb,
            baselines=baseline,
            additional_forward_args=(attn,),
            target=pred_cls,
            n_steps=20,
            return_convergence_delta=True
        )
        attn_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        toks = tok.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        s = attn_scores
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)

        results.append({
            "text": text,
            "tokens": toks,
            "attr": s.tolist(),
            "pred_idx": pred_cls,
            "probs": probs.tolist()
        })
    return results

def _lime_explain(texts, tok, model):
    try:
        from lime.lime_text import LimeTextExplainer
    except Exception as e:
        print(f"[WARN] LIME not available: {e}")
        return []

    def predict_proba(strs):
        return _predict_proba(strs, tok, model)

    explainer = LimeTextExplainer(class_names=None)
    results = []
    for text in texts:
        exp = explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_proba,
            num_features=15
        )
        probs = _predict_proba([text], tok, model)[0]
        pred = int(np.argmax(probs))
        weights = dict(exp.as_list(label=pred))
        # crude tokenization for display; LIME renders on words
        toks = re.findall(r"\w+|\W", text)[:MAX_TOKENS]
        scores = [weights.get(t, 0.0) for t in toks]
        s = np.array(scores, dtype=float)
        if len(s):
            s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        results.append({
            "text": text,
            "tokens": toks,
            "attr": s.tolist(),
            "pred_idx": pred,
            "probs": probs.tolist()
        })
    return results

def _attention_rollout(texts, tok, model):
    import torch
    results = []
    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS, add_special_tokens=True)
        with torch.no_grad():
            out = model(**enc, output_attentions=True)
            logits = out.logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            pred = int(torch.argmax(logits, dim=-1).item())
            atts = out.attentions  # tuple of L tensors [B,H,T,T]

        # rollout: avg heads per layer, add residual, row-normalize, multiply through layers
        A = None
        for layer_att in atts:
            a = layer_att.mean(dim=1).squeeze(0)  # (T,T)
            eye = torch.eye(a.size(-1), device=a.device, dtype=a.dtype)
            a = a + eye
            a = a / a.sum(dim=-1, keepdim=True)
            A = a if A is None else A @ a

        # importance to [CLS] (token 0): contributions from tokens → CLS
        imp = A[:, 0].detach().cpu().numpy()
        imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)
        toks = tok.convert_ids_to_tokens(enc["input_ids"].squeeze(0).tolist())

        results.append({
            "text": text,
            "tokens": toks,
            "attr": imp.tolist(),
            "pred_idx": pred,
            "probs": probs.tolist()
        })
    return results

# ---------------------------
# Faithfulness (deletion)
# ---------------------------
def _detok_from_wordpieces(tokens):
    """Very simple WordPiece detokenizer for deletion tests."""
    out = []
    for t in tokens:
        if t in ("[CLS]", "[SEP]", "[PAD]", "[UNK]"):
            continue
        if t.startswith("##") and out:
            out[-1] = out[-1] + t[2:]
        else:
            out.append(t)
    return out

def _faithfulness_curves(method_name, examples, tok, model):
    """
    Deletion test: remove top-k tokens progressively and track predicted prob of original class.
    This uses a simple WordPiece detokenization; it's approximate but useful for sanity checks.
    """
    from transformers import TextClassificationPipeline
    try:
        pipe = TextClassificationPipeline(model=model, tokenizer=tok, return_all_scores=True, truncation=True)
    except Exception:
        return pd.DataFrame(columns=["method","frac_removed","prob","example_idx"])

    rows = []
    for i, ex in enumerate(examples):
        toks_wp = ex["tokens"]
        attr = np.array(ex["attr"], dtype=float)
        pred_idx = int(ex["pred_idx"])

        # map to rough words for readable deletion; align lengths if mismatch
        toks = _detok_from_wordpieces(toks_wp)
        if len(toks) < 12:
            continue

        # simple attribution downsampling to word level (mean over contiguous pieces)
        # (fallback: if lengths mismatch badly, just use word-level uniform attributions)
        if len(attr) == len(toks_wp):
            # group by merging '##' pieces; build mapping
            word_attr = []
            acc = 0.0
            count = 0
            for t, a in zip(toks_wp, attr):
                if t in ("[CLS]", "[SEP]", "[PAD]", "[UNK]"):
                    continue
                if t.startswith("##"):
                    acc += a; count += 1
                else:
                    if count > 0:
                        word_attr.append(acc / max(1, count))
                    acc = a; count = 1
            if count > 0:
                word_attr.append(acc / max(1, count))
            wa = np.array(word_attr, dtype=float)
            if len(wa) != len(toks):
                wa = np.interp(np.linspace(0, len(wa)-1, num=len(toks)), np.arange(len(wa)), wa)
        else:
            wa = np.ones(len(toks), dtype=float)

        order = np.argsort(wa)[::-1]  # high → low
        for frac in [0, 0.05, 0.1, 0.2, 0.3, 0.5]:
            k = max(1, int(round(frac * len(toks))))
            keep_mask = np.ones(len(toks), dtype=bool)
            keep_mask[order[:k]] = False
            pruned_text = " ".join([t for t, keep in zip(toks, keep_mask) if keep])
            try:
                sc = pipe([pruned_text])[0][pred_idx]["score"]
            except Exception:
                sc = np.nan
            rows.append({"method": method_name, "frac_removed": frac, "prob": sc, "example_idx": i})
    return pd.DataFrame(rows)

# ---------------------------
# Main
# ---------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR_IG.mkdir(parents=True, exist_ok=True)
    OUT_DIR_LIME.mkdir(parents=True, exist_ok=True)
    OUT_DIR_ATTN.mkdir(parents=True, exist_ok=True)

    df = _load_samples()
    if df is None or df.empty:
        return

    tok, model, id2label = _prep_hf()
    if tok is None:
        return

    texts = df["text"].tolist()

    # Predictions table
    try:
        probs = _predict_proba(texts, tok, model, batch_size=16)
        preds = probs.argmax(axis=1)
        ex = pd.DataFrame({
            "text": texts,
            "pred_idx": preds,
            "pred_label": [id2label.get(int(i), str(int(i))) for i in preds],
            "p_neg": probs[:,0] if probs.shape[1] > 0 else np.nan,
            "p_neu": probs[:,1] if probs.shape[1] > 1 else np.nan,
            "p_pos": probs[:,2] if probs.shape[1] > 2 else np.nan,
        })
        ex.to_csv(OUT_DIR / "examples.csv", index=False)
        (OUT_DIR / "meta.json").write_text(json.dumps({"model": MODEL_NAME, "id2label": id2label}, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Prediction table failed: {e}")

    # IG
    ig_res = _integrated_gradients(texts, tok, model, id2label)
    if ig_res:
        with (OUT_DIR_IG / "attributions.jsonl").open("w", encoding="utf-8") as f:
            for item in ig_res:
                f.write(json.dumps(item) + "\n")

    # LIME
    lime_res = _lime_explain(texts, tok, model)
    if lime_res:
        with (OUT_DIR_LIME / "attributions.jsonl").open("w", encoding="utf-8") as f:
            for item in lime_res:
                f.write(json.dumps(item) + "\n")

    # Attention rollout
    attn_res = _attention_rollout(texts, tok, model)
    if attn_res:
        with (OUT_DIR_ATTN / "attributions.jsonl").open("w", encoding="utf-8") as f:
            for item in attn_res:
                f.write(json.dumps(item) + "\n")

    # Faithfulness curves
    frames = []
    if ig_res:   frames.append(_faithfulness_curves("IG", ig_res, tok, model))
    if lime_res: frames.append(_faithfulness_curves("LIME", lime_res, tok, model))
    if attn_res: frames.append(_faithfulness_curves("ATTN", attn_res, tok, model))
    if frames:
        curves = pd.concat(frames, ignore_index=True)
        curves.to_csv(OUT_DIR / "faithfulness_curves.csv", index=False)

    # Human-friendly markdown cards
    try:
        def _mk_card(items, name):
            lines = [f"## {name} examples\n"]
            for i, ex in enumerate(items[:10]):
                toks = ex["tokens"]; attr = np.array(ex["attr"])
                top = np.argsort(attr)[::-1][:8]
                highlights = [toks[j] for j in top]
                snippet = (ex["text"][:240] + "…").replace("\n", " ")
                lines.append(f"**Example {i+1}**  \nText: {snippet}  \nTop tokens: `{', '.join(highlights)}`  \nPred idx: {ex['pred_idx']}")
            return "\n".join(lines)

        cards = []
        if ig_res:   cards.append(_mk_card(ig_res, "Integrated Gradients"))
        if lime_res: cards.append(_mk_card(lime_res, "LIME"))
        if attn_res: cards.append(_mk_card(attn_res, "Attention Rollout"))
        (OUT_DIR / "example_cards.md").write_text("\n\n---\n\n".join(cards), encoding="utf-8")
    except Exception:
        pass

    print(f"[DONE] Deep XAI artifacts → {OUT_DIR}")

if __name__ == "__main__":
    main()
