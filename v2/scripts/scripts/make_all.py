# make_all.py
"""
One-click orchestrator for the full pipeline.

- Runs scripts in dependency-aware order.
- Tees stdout/stderr to outputs/make_all.log.
- Skips gracefully if a script file is missing.
- Writes a JSON summary and paper note snippets.

Run: python make_all.py
"""

from __future__ import annotations
import io
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
import runpy

# ---- Paths (use the same ROOT as scripts/_config.py) ----
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"

# Best effort import of _config for OUTPUTS_DIR, but don't fail if missing.
try:
    sys.path.insert(0, str(SCRIPTS))
    import _config as CFG  # type: ignore
    ROOT = CFG.ROOT
    OUTPUTS_DIR = CFG.OUTPUTS_DIR
except Exception:
    ROOT = HERE
    OUTPUTS_DIR = HERE / "outputs"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUTPUTS_DIR / "make_all.log"
SUMMARY_JSON = OUTPUTS_DIR / "make_summary.json"
PAPER_NOTES = OUTPUTS_DIR / "paper_notes.txt"

# ---- Tee logger ------------------------------------------------------------
class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)
    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

def open_log() -> SimpleNamespace:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    f = LOG_PATH.open("a", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = Tee(old_out, f)
    sys.stderr = Tee(old_err, f)
    return SimpleNamespace(file=f, old_out=old_out, old_err=old_err)

def close_log(h):
    try:
        sys.stdout = h.old_out
        sys.stderr = h.old_err
    finally:
        try:
            h.file.close()
        except Exception:
            pass

# ---- Runner ----------------------------------------------------------------
def run_script(rel_name: str) -> dict:
    """
    Execute a script with runpy so it shares the same interpreter & env.
    Returns a dict with {status, started, ended, seconds, error}
    """
    path = SCRIPTS / rel_name
    rec = {"script": rel_name, "status": "skipped", "started": None, "ended": None, "seconds": 0.0, "error": None}
    if not path.exists():
        print(f"[SKIP] {rel_name} (file not found)")
        return rec
    print(f"\n===== RUN {rel_name} =====")
    rec["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()
    try:
        runpy.run_path(str(path), run_name="__main__")
        rec["status"] = "ok"
    except SystemExit as e:
        # allow scripts that call sys.exit(0/1)
        code = getattr(e, "code", 0)
        if code in (0, None):
            rec["status"] = "ok"
        else:
            rec["status"] = "error"
            rec["error"] = f"SystemExit({code})"
            print(f"[ERROR] {rel_name} exited with code {code}")
    except Exception:
        rec["status"] = "error"
        rec["error"] = traceback.format_exc()
        print(f"[ERROR] Exception while running {rel_name}:\n{rec['error']}")
    finally:
        rec["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        rec["seconds"] = round(time.time() - t0, 2)
        print(f"===== END {rel_name} [{rec['status']}] in {rec['seconds']}s =====\n")
    return rec

# ---- Planned order ---------------------------------------------------------
# We run leakage guard *early* (after ingest + benches) so training can exclude GOLD safely.
ORDER = [
    # Batch 1
    "00_setup.py",
    "01_fetch_benchmarks.py",
    "02_ingest_marketaux.py",
    # Early guard before any training/eval
    "15_data_leakage_guard.py",
    "02b_make_annotation_batches.py",

    # Batch 2 · Baselines
    "03_build_lm_lists.py",
    "04_lexicon_baselines.py",
    "05_lr_shap_benchmarks.py",
    "05b_ebm_gam_benchmarks.py",
    "06_finbert_benchmarks.py",

    # Batch 3 · Gold evaluation & multilingual
    "07_marketaux_weaklabel_lr.py",
    "07b_xlm_infer_marketaux.py",
    "08_eval_on_manual_annotations.py",
    "09_plot_confmats.py",

    # Batch 4 · Regimes & weak-label bias
    "10_shap_regime_shift_all.py",
    "10_weak_label_bias_audit.py",

    # Batch 5 · Econometrics & strategies
    "11_event_study.py",
    "12_granger_var.py",
    "13_backtest_sector_strategies.py",

    # Batch 6 · Deep XAI, hybrid, diagram (guard already ran)
    "16_deep_explain_finbert.py",
    "17_hybrid_embeddings_lr.py",
    "18_workflow_diagram.py",
]

# ---- Paper notes (tone & structure) ----------------------------------------
NOTES = """\
# Paper Notes (pasteable snippets)

## Abstract (tone-down & clarity)
- We present a *reproducible* and *transparent* workflow to study the accuracy–interpretability trade-off in financial news sentiment across **10 industries** and **multi-lingual** sources.
- Results are **modest but consistent**: transformer models (FinBERT) achieve the best macro-F1 on our new gold set, while glass-box models remain competitive and provide clearer explanations.
- A **weak-label bias audit** quantifies errors introduced by VADER and shows meaningful divergence in explanations versus human-labeled models.
- Proper econometric tests (**event studies** and **Granger causality**) find **limited and regime-dependent** predictive content.
- Contributions: (i) open pipeline & code, (ii) cross-industry, multilingual evaluation corpus, (iii) bias audit for weak supervision, (iv) black-box and glass-box explainability, (v) econometric validation beyond naive correlations.

## Introduction (bridge to results)
- Preview key findings briefly: FinBERT > LR/lexicons in accuracy; interpretable models help analysts understand drivers; predictive signals for ETFs are weak overall, stronger in select regimes; weak supervision adds measurable bias.
- Position novelty in **scope and rigor**, not algorithms alone: cross-industry + multilingual + bias audit + econometrics + XAI for both simple and deep models.

## Methods (what’s new)
- Add a **leakage guard** (§3.x) ensuring disjointness between temporal corpus, gold test, and benchmarks (URL hash + simhash).
- Extend XAI to **black-box** (FinBERT) with **Integrated Gradients, LIME, attention rollout**, and **faithfulness** (deletion curves).
- Include **glass-box** EBM/GAM and a **hybrid** (FinBERT-CLS → LR/EBM with SHAP).

## Results (calibrated claims)
- Report macro-F1/acc with CIs; emphasize **cross-industry** performance and **language slices**.
- Show **bias audit**: label error vs. gold, SHAP rank correlations, attribution divergence.
- Replace correlation tables with **event-study** abnormal returns and **Granger** p-values; highlight limited, regime-dependent predictability.
- Include confusion matrices and a **workflow figure**.

## Discussion (moderate tone)
- Frame “simpler models in regulated settings” as a **hypothesis** supported by interpretability/faithfulness advantages—not as a universal rule.
- Acknowledge limitations: dataset shift, annotation size/noise, non-EN coverage variance, and constraints of daily ETF proxies.
- Connect future work directly to weaknesses (larger multilingual gold set, counterfactuals, RAG-XAI, multi-modal fusion).

## Conclusion (realistic)
- This is a **foundation**: a documented pipeline and evaluation protocol; not a plug-and-play trading system.
- Code and artifacts enable **replication and extension** by the community.

## Citations to add
- Include the fintech reference suggested by Reviewer 1 (DOI: 10.1007/978-981-99-3300-6_5).
"""

def write_notes():
    PAPER_NOTES.parent.mkdir(parents=True, exist_ok=True)
    PAPER_NOTES.write_text(NOTES, encoding="utf-8")
    print(f"[INFO] Wrote paper notes → {PAPER_NOTES}")

# ---- Main ------------------------------------------------------------------
def main():
    h = open_log()
    print("========== MAKE ALL START ==========")
    print(f"Root: {ROOT}")
    print(f"Scripts dir: {SCRIPTS.relative_to(ROOT) if SCRIPTS.exists() else SCRIPTS}")
    print(f"Log: {LOG_PATH.relative_to(ROOT) if LOG_PATH.exists() else LOG_PATH}")

    summary = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": [],
        "ended": None,
        "duration_sec": None,
        "ok": 0,
        "error": 0,
        "skipped": 0,
    }

    t0 = time.time()
    for rel in ORDER:
        rec = run_script(rel)
        summary["steps"].append(rec)
        summary[rec["status"]] = summary.get(rec["status"], 0) + 1

    write_notes()

    summary["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["duration_sec"] = round(time.time() - t0, 2)

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary → {SUMMARY_JSON}")

    print("=========== MAKE ALL END ===========")
    close_log(h)

if __name__ == "__main__":
    main()
