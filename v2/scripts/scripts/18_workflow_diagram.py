# scripts/18_workflow_diagram.py
"""
Workflow Diagram (Graphviz → PNG; ASCII fallback)

Outputs:
  outputs/figure_workflow.png          (if graphviz available)
  outputs/figure_workflow_ascii.txt    (always)
Run: python scripts/18_workflow_diagram.py
"""
from pathlib import Path

from _config import OUTPUTS_DIR

OUT_PNG = OUTPUTS_DIR / "figure_workflow.png"
OUT_TXT = OUTPUTS_DIR / "figure_workflow_ascii.txt"

ASCII = r"""
RAW
├─ data/raw/marketaux/news.json
│
├─ 02_ingest_marketaux.py → data/processed/marketaux/*.csv
│    └─ 02b_make_annotation_batches.py → data/annotation/batches/*.csv
│         └─ (human) annotate → data/annotation/annotated_articles.csv (GOLD)
│
├─ 01_fetch_benchmarks.py → FPB.csv, FiQA_headlines.csv
│
├─ 15_data_leakage_guard.py → outputs/leakage/*  [EXCLUDE GOLD FROM TRAIN]
│
├─ Baselines (EN):
│   ├─ 04_lexicon_baselines.py
│   ├─ 05_lr_shap_benchmarks.py
│   ├─ 05b_ebm_gam_benchmarks.py
│   └─ 06_finbert_benchmarks.py
│
├─ 07_marketaux_weaklabel_lr.py (all industries) → outputs/weaklabel_lr/*
│
├─ 07b_xlm_infer_marketaux.py (non-EN) → outputs/xlm_infer_*.*
│
├─ 08_eval_on_manual_annotations.py → outputs/eval/*
│
├─ 09_plot_confmats.py → outputs/fig_confmats/*
│
├─ 10_shap_regime_shift_all.py → outputs/regime/*
│   └─ 10_weak_label_bias_audit.py → outputs/bias_audit/*
│
├─ Deep XAI & Hybrid:
│   ├─ 16_deep_explain_finbert.py → outputs/deep_xai/*
│   └─ 17_hybrid_embeddings_lr.py → outputs/hybrid/*
│
└─ Econometrics & Strategies:
    ├─ 11_event_study.py → outputs/event_study/*
    ├─ 12_granger_var.py → outputs/granger/*
    └─ 13_backtest_sector_strategies.py → outputs/backtest/*
"""

def _graphviz_png():
    try:
        from graphviz import Digraph
    except Exception as e:
        return False

    g = Digraph("workflow", format="png")
    g.attr(rankdir="LR", nodesep="0.4", ranksep="0.5", fontsize="10")

    # Nodes
    g.node("raw", "data/raw/marketaux/news.json", shape="folder")
    g.node("ing", "02_ingest_marketaux.py\n→ processed CSVs", shape="box")
    g.node("batches", "02b_make_annotation_batches.py", shape="box")
    g.node("gold", "annotated_articles.csv (GOLD)", shape="component")
    g.node("bench", "01_fetch_benchmarks.py\n→ FPB/FiQA", shape="box")
    g.node("guard", "15_data_leakage_guard.py\n(EXCLUDE GOLD from train)", shape="octagon")

    g.node("base", "Baselines (EN):\n04 lexicons\n05 LR+SHAP\n05b EBM/GAM\n06 FinBERT", shape="box")
    g.node("weak", "07 weak-label LR\n(all industries)", shape="box")
    g.node("xlm", "07b XLM-R (non-EN)", shape="box")
    g.node("eval", "08 eval on GOLD", shape="box")
    g.node("conf", "09 confusion matrices", shape="box")
    g.node("reg", "10 SHAP regime shift", shape="box")
    g.node("bias", "10 weak-label bias audit", shape="box")
    g.node("deep", "16 deep explain FinBERT", shape="box")
    g.node("hyb", "17 hybrid (CLS→LR/EBM)", shape="box")
    g.node("evt", "11 event study", shape="box")
    g.node("gr", "12 granger/VAR", shape="box")
    g.node("bt", "13 backtests", shape="box")

    # Edges
    g.edge("raw", "ing")
    g.edge("ing", "batches")
    g.edge("batches", "gold")
    g.edge("ing", "weak")
    g.edge("raw", "bench")
    g.edge("gold", "guard")
    g.edge("ing", "guard")
    g.edge("bench", "base")
    g.edge("ing", "base")
    g.edge("guard", "weak")
    g.edge("weak", "eval")
    g.edge("gold", "eval")
    g.edge("eval", "conf")
    g.edge("weak", "reg")
    g.edge("gold", "bias")
    g.edge("weak", "bias")
    g.edge("base", "deep")
    g.edge("base", "hyb")
    g.edge("weak", "evt")
    g.edge("weak", "gr")
    g.edge("weak", "bt")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    g.render(filename=str(OUT_PNG.with_suffix("")), cleanup=True)
    return True

def main():
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(ASCII.strip()+"\n", encoding="utf-8")
    ok = _graphviz_png()
    if ok:
        print(f"[DONE] Workflow figure → {OUT_PNG}")
    else:
        print(f"[OK] Wrote ASCII workflow → {OUT_TXT}  (install `graphviz` to get PNG)")

if __name__ == "__main__":
    main()
