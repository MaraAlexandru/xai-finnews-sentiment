# scripts/03_run_finbert.py
"""
FinBERT benchmarks on datasets with columns: text,y (y in {0=neg, 1=neu, 2=pos})

Modes:
  --mode inference   → zero-shot predictions (no training)
  --mode finetune    → fine-tune on an 80/20 stratified split

Designed for GTX 1050 Ti (4GB VRAM) + Ryzen 5 3600 + Windows:
- Auto CPU/GPU detection (works even if PyTorch is CPU-only)
- AMP (fp16) shims for old/new PyTorch (no deprecation warnings)
- Pre-tokenize once in the main process (Windows-safe workers)
- Top-level collator class (picklable) + dynamic padding per batch
- Optional gradient checkpointing for tight VRAM
- Progress bars (tqdm) for pretokenize, train, eval, and inference
- If workers fail on Windows, auto-falls back to num_workers=0

Outputs (paper-ready):
  inference:
    outputs/finbert_infer_{name}.json
    outputs/finbert_infer_report_{name}.txt
    outputs/finbert_infer_confmat_{name}.csv
  finetune:
    models/finbert_{name}_manual/
    models/finbert_{name}_manual/eval_results.json
    outputs/finbert_ft_report_{name}.txt
    outputs/finbert_ft_confmat_{name}.csv
    outputs/finbert_ft_preds_{name}.csv
"""

import argparse, json, os, sys, platform, contextlib
from pathlib import Path

import numpy as np
import pandas as pd

from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, set_seed
)

# ---------------- QoL: small speed gain on PyTorch 2.x (no-op if unsupported) ----------------
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ---------------- AMP shims: autocast + GradScaler, old/new PyTorch, no warnings -------------
def _autocast_ctx(enabled: bool, device_type: str = "cuda"):
    if not enabled or device_type != "cuda":
        return contextlib.nullcontext()
    try:
        from torch.amp import autocast as _autocast_new  # PyTorch ≥ 2.0
        return _autocast_new(device_type=device_type, enabled=True)
    except Exception:
        from torch.cuda.amp import autocast as _autocast_old  # PyTorch 1.x
        return _autocast_old(enabled=True)

class _DummyScaler:
    def __init__(self): pass
    def is_enabled(self): return False
    def scale(self, x): return x
    def step(self, opt): opt.step()
    def update(self): pass
    def unscale_(self, opt): pass

def _make_grad_scaler(enabled: bool, device_type: str = "cuda"):
    if not enabled or device_type != "cuda":
        return _DummyScaler()
    try:
        from torch.amp import GradScaler as _GradScalerNew  # PyTorch ≥ 2.0
        return _GradScalerNew(device_type=device_type)
    except Exception:
        from torch.cuda.amp import GradScaler as _GradScalerOld  # PyTorch 1.x
        return _GradScalerOld(enabled=True)

# ---------------- general helpers -----------------------------------------------------------
TARGET_ORDER = ["negative", "neutral", "positive"]  # our y: 0,1,2

def build_label_mapping(id2label: dict) -> np.ndarray:
    id2lab = {int(k): str(v).lower() for k, v in id2label.items()}
    synonyms = {"negative": {"negative", "neg"},
                "neutral":  {"neutral", "neu"},
                "positive": {"positive", "pos"}}
    idx_map = []
    for target in TARGET_ORDER:
        found = None
        for midx, mlabel in id2lab.items():
            if mlabel in synonyms[target]:
                found = midx; break
        if found is None:
            for midx, mlabel in id2lab.items():
                if target in mlabel:
                    found = midx; break
        if found is None:
            raise ValueError(f"Cannot map labels {id2lab} -> {TARGET_ORDER}")
        idx_map.append(found)
    return np.array(idx_map, dtype=int)

def reorder_logits(logits: np.ndarray, idx_map: np.ndarray) -> np.ndarray:
    return logits[:, idx_map]

def save_report_confmat(prefix: str, y_true: np.ndarray, y_pred: np.ndarray):
    Path("outputs").mkdir(exist_ok=True)
    (Path("outputs") / f"{prefix}_report.txt").write_text(
        classification_report(y_true, y_pred, digits=3)
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    pd.DataFrame(cm, index=["true_neg","true_neu","true_pos"],
                 columns=["pred_neg","pred_neu","pred_pos"]
    ).to_csv(Path("outputs") / f"{prefix}_confmat.csv", index=True)

def _make_stratified_datasetdict(df: pd.DataFrame, seed=42) -> DatasetDict:
    Xtr, Xte, ytr, yte = train_test_split(
        df["text"].astype(str), df["y"].astype(int),
        test_size=0.2, stratify=df["y"].astype(int), random_state=seed
    )
    dtrain = Dataset.from_pandas(pd.DataFrame({"text": Xtr.values, "labels": ytr.values}))
    dtest  = Dataset.from_pandas(pd.DataFrame({"text": Xte.values, "labels": yte.values}))
    return DatasetDict(train=dtrain, test=dtest)

def pick_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ---------------- zero-shot inference (progress bars + AMP + dynamic padding) ---------------
def run_inference(model_id: str, df: pd.DataFrame, name: str,
                  fp16: bool = True, max_length: int = 96, batch_size: int = 64,
                  force_cpu: bool = False, pbar: bool = True):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    idx_map = build_label_mapping(mdl.config.id2label)

    texts = df["text"].astype(str).tolist()
    y     = df["y"].astype(int).values

    device = pick_device(force_cpu=force_cpu)
    mdl.to(device); mdl.eval()
    torch.backends.cudnn.benchmark = (device.type == "cuda")

    preds, prob_rows = [], []
    it = range(0, len(texts), batch_size)
    it = tqdm(it, desc="Inference", unit="batch", leave=False) if pbar else it

    for i in it:
        enc = tok(texts[i:i+batch_size], truncation=True, padding=True,
                  max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad(), _autocast_ctx(fp16 and device.type == "cuda", "cuda"):
            logits = mdl(**enc).logits
        logits = logits.detach().cpu().numpy()
        logits = reorder_logits(logits, idx_map)
        prob = torch.tensor(logits).softmax(dim=1).numpy()
        preds.append(prob.argmax(axis=1)); prob_rows.append(prob)

    y_pred = np.concatenate(preds)
    acc = accuracy_score(y, y_pred)
    mf1 = f1_score(y, y_pred, average="macro")

    save_report_confmat(f"finbert_infer_{name}", y, y_pred)
    (Path("outputs") / f"finbert_infer_{name}.json").write_text(json.dumps({
        "dataset": name, "model": model_id, "mode": "inference",
        "n": int(len(df)), "accuracy": float(acc), "macro_f1": float(mf1),
        "counts": {"neg": int((y==0).sum()), "neu": int((y==1).sum()), "pos": int((y==2).sum())}
    }, indent=2))
    print(f"[Zero-shot] {model_id} on {name}: Acc={acc:.3f}  Macro-F1={mf1:.3f}  (device={device})")

# ---------------- pre-tokenize once (Windows-safe) with progress bar ------------------------
def pretokenize_texts(tokenizer, texts, max_length: int, batch_size: int = 512, pbar: bool = True):
    ids, masks = [], []
    it = range(0, len(texts), batch_size)
    it = tqdm(it, desc="Pretokenize", unit="batch", leave=False) if pbar else it
    for i in it:
        chunk = list(texts[i:i+batch_size])
        enc = tokenizer(chunk, truncation=True, padding=False, max_length=max_length,
                        return_attention_mask=True)
        ids.extend(enc["input_ids"])
        masks.extend(enc["attention_mask"])
    return ids, masks

class PreTokenizedDataset(TorchDataset):
    def __init__(self, input_ids, attn_mask, labels):
        self.input_ids = list(input_ids)
        self.attn_mask = list(attn_mask)
        self.labels = list(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attn_mask[idx], int(self.labels[idx]))

# ---------------- top-level collator class (picklable) --------------------------------------
class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
    def __call__(self, batch):
        ids_list, mask_list, labels = zip(*batch)
        L = max(len(x) for x in ids_list)
        def pad(seq, pad_val, L_):
            return seq[:L_] if len(seq) >= L_ else seq + [pad_val]*(L_ - len(seq))
        input_ids = [pad(seq, self.pad_token_id, L) for seq in ids_list]
        attn_mask = [pad(seq, 0,                  L) for seq in mask_list]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# ---------------- simple LR schedule w/ warmup ----------------------------------------------
def linear_schedule(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        return max(0.0, float(num_training_steps - step) / max(1, num_training_steps - num_warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ---------------- manual fine-tune loop (fast + robust + progress bars) ---------------------
def finetune_manual_loop(model_id: str, df: pd.DataFrame, name: str,
                         epochs: int = 2, lr: float = 2e-5, batch_size: int = 24,
                         seed: int = 42, max_length: int = 96,
                         num_workers: int = 4, fp16: bool = True,
                         grad_ckpt: bool = False, grad_accum_steps: int = 1,
                         force_cpu: bool = False, weight_decay: float = 0.01,
                         warmup_ratio: float = 0.1, pbar: bool = True):
    set_seed(seed)
    device = pick_device(force_cpu=force_cpu)
    torch.backends.cudnn.benchmark = (device.type == "cuda")

    # Split
    Xtr, Xte, ytr, yte = train_test_split(
        df["text"].astype(str), df["y"].astype(int),
        test_size=0.2, stratify=df["y"].astype(int), random_state=seed
    )

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id if tok.eos_token_id is not None else 0)

    # Pre-tokenize ONCE (Windows-safe)
    tr_ids, tr_mask = pretokenize_texts(tok, Xtr.values, max_length=max_length, pbar=pbar)
    te_ids, te_mask = pretokenize_texts(tok, Xte.values, max_length=max_length, pbar=pbar)

    tr_ds = PreTokenizedDataset(tr_ids, tr_mask, ytr.values)
    te_ds = PreTokenizedDataset(te_ids, te_mask, yte.values)

    pin = (device.type == "cuda")
    persistent = (num_workers > 0 and not platform.system().lower().startswith("win"))
    collator = PadCollator(pad_id)

    # Try workers; fall back to 0 on Windows issues
    try:
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=pin,
                               persistent_workers=persistent,
                               prefetch_factor=(2 if num_workers > 0 else None),
                               collate_fn=collator)
        _ = next(iter(tr_loader))
    except Exception as e:
        print(f"[Info] DataLoader workers failed ({repr(e)}). Falling back to num_workers=0.")
        num_workers = 0
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=pin,
                               collate_fn=collator)

    try:
        te_loader = DataLoader(te_ds, batch_size=max(2, batch_size*2), shuffle=False,
                               num_workers=num_workers, pin_memory=pin,
                               persistent_workers=persistent,
                               prefetch_factor=(2 if num_workers > 0 else None),
                               collate_fn=collator)
        _ = next(iter(te_loader))
    except Exception as e:
        print(f"[Info] Eval DataLoader workers failed ({repr(e)}). Falling back to num_workers=0.")
        te_loader = DataLoader(te_ds, batch_size=max(2, batch_size*2), shuffle=False,
                               num_workers=0, pin_memory=pin,
                               collate_fn=collator)

    mdl = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)
    mdl.config.id2label = {0:"negative",1:"neutral",2:"positive"}
    mdl.config.label2id = {"negative":0,"neutral":1,"positive":2}
    if grad_ckpt and hasattr(mdl, "gradient_checkpointing_enable"):
        if hasattr(mdl.config, "use_cache"): mdl.config.use_cache = False
        mdl.gradient_checkpointing_enable()
    mdl.to(device); mdl.train()

    optim = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = max(1, (len(tr_loader) * epochs) // max(1, grad_accum_steps))
    sched = linear_schedule(optim, int(warmup_ratio * total_steps), total_steps)

    scaler = _make_grad_scaler(fp16 and device.type == "cuda", device_type="cuda")

    best_mf1, best_state = -1.0, None
    global_step = 0

    for ep in range(1, epochs + 1):
        running = 0.0
        optim.zero_grad(set_to_none=True)
        iter_train = tqdm(tr_loader, desc=f"Train epoch {ep}/{epochs}", unit="batch", leave=False) if pbar else tr_loader
        for step, batch in enumerate(iter_train, start=1):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with _autocast_ctx(fp16 and device.type == "cuda", "cuda"):
                out = mdl(**batch)
                loss = out.loss / max(1, grad_accum_steps)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % max(1, grad_accum_steps) == 0:
                if scaler.is_enabled():
                    scaler.step(optim); scaler.update()
                else:
                    optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
            running += loss.item() * max(1, grad_accum_steps)
            if pbar:
                iter_train.set_postfix(loss=f"{running/step:.4f}")

        # ---- quick eval each epoch ----
        mdl.eval()
        preds = []
        iter_eval = tqdm(te_loader, desc=f"Eval epoch {ep}/{epochs}", unit="batch", leave=False) if pbar else te_loader
        with torch.no_grad():
            for batch in iter_eval:
                batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                with _autocast_ctx(fp16 and device.type == "cuda", "cuda"):
                    logits = mdl(**batch).logits
                preds.append(logits.detach().cpu().numpy())
        logits = np.vstack(preds)
        y_pred = logits.argmax(axis=1)
        acc = accuracy_score(yte.values, y_pred)
        mf1 = f1_score(yte.values, y_pred, average="macro")
        print(f"[Manual] epoch {ep}/{epochs}  train_loss={running/len(tr_loader):.4f}  val_acc={acc:.3f}  val_macroF1={mf1:.3f}")
        if mf1 > best_mf1:
            best_mf1 = mf1
            best_state = {k: v.cpu().clone() for k, v in mdl.state_dict().items()}
        mdl.train()

    if best_state is not None:
        mdl.load_state_dict(best_state)

    # final eval + save artifacts
    mdl.eval()
    preds = []
    iter_final = tqdm(te_loader, desc="Final eval", unit="batch", leave=False) if pbar else te_loader
    with torch.no_grad():
        for batch in iter_final:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with _autocast_ctx(fp16 and device.type == "cuda", "cuda"):
                logits = mdl(**batch).logits
            preds.append(logits.detach().cpu().numpy())
    logits = np.vstack(preds)
    y_pred = logits.argmax(axis=1)

    acc = accuracy_score(yte.values, y_pred)
    mf1 = f1_score(yte.values, y_pred, average="macro")
    print(f"[Manual-FINAL] {model_id} on {name}: Acc={acc:.3f}  Macro-F1={mf1:.3f}  (device={device})")

    out_dir = Path(f"models/finbert_{name}_manual"); out_dir.mkdir(parents=True, exist_ok=True)
    mdl.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "eval_results.json").write_text(json.dumps({"accuracy": float(acc), "macro_f1": float(mf1)}, indent=2))

    save_report_confmat(f"finbert_ft_{name}", yte.values, y_pred)
    pd.DataFrame({"text": Xte.values, "gold_y": yte.values, "pred_y": y_pred}).to_csv(
        Path("outputs") / f"finbert_ft_preds_{name}.csv", index=False
    )

# ---------------- CLI -----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV with columns text,y")
    ap.add_argument("--model", default="ProsusAI/finbert", help="HF model ID")
    ap.add_argument("--mode", choices=["inference", "finetune"], default="inference")

    # common
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_cpu", type=int, default=0)     # set 1 to force CPU
    ap.add_argument("--pbar", type=int, default=1)          # 1=show progress bars

    # inference knobs
    ap.add_argument("--inf_batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=96)
    ap.add_argument("--fp16", type=int, default=1)          # 1=True, 0=False

    # finetune knobs
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--num_workers", type=int, default=4)   # auto-fallback to 0 if Windows balks
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)

    args = ap.parse_args()

    Path("outputs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    df = pd.read_csv(args.csv).dropna(subset=["text", "y"])
    name = Path(args.csv).stem
    show_pbar = bool(args.pbar)

    if args.mode == "inference":
        run_inference(
            args.model, df, name,
            fp16=bool(args.fp16),
            max_length=args.max_length,
            batch_size=args.inf_batch_size,
            force_cpu=bool(args.force_cpu),
            pbar=show_pbar,
        )
    else:
        finetune_manual_loop(
            args.model, df, name,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
            max_length=args.max_length, num_workers=args.num_workers, fp16=bool(args.fp16),
            grad_ckpt=bool(args.grad_ckpt), grad_accum_steps=args.grad_accum,
            force_cpu=bool(args.force_cpu),
            weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
            pbar=show_pbar
        )
