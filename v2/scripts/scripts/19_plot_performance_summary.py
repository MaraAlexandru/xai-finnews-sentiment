# scripts/19_plot_performance_summary.py
"""
Generates a summary bar chart of model performance (Macro F1)
across all key datasets.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# --- Configuration ---
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
FIGURE_PATH = OUTPUT_DIR / "figures" / "fig_performance_summary.png"

# --- Data Loading and Parsing ---
def get_metrics():
    # Public Benchmarks
    lexicon_df = pd.read_csv(OUTPUT_DIR / "baselines" / "lexicon_metrics.csv")
    lr_fpb = json.loads((OUTPUT_DIR / "baselines" / "lr_shap" / "fpb_metrics.json").read_text())
    lr_fiqa = json.loads((OUTPUT_DIR / "baselines" / "lr_shap" / "fiqa_metrics.json").read_text())
    ebm_fpb = json.loads((OUTPUT_DIR / "baselines" / "ebm_gam" / "fpb_metrics.json").read_text())
    ebm_fiqa = json.loads((OUTPUT_DIR / "baselines" / "ebm_gam" / "fiqa_metrics.json").read_text())
    hybrid_fpb = json.loads((OUTPUT_DIR / "hybrid" / "fpb" / "metrics.json").read_text())
    hybrid_fiqa = json.loads((OUTPUT_DIR / "hybrid" / "fiqa" / "metrics.json").read_text())

    # Gold Standard
    gold_df = pd.read_csv(OUTPUT_DIR / "gold_eval" / "global_metrics.csv")

    data = [
        {'model': 'VADER', 'dataset': 'FPB', 'macro_f1': lexicon_df.query("model=='VADER' and dataset=='fpb'")['macro_f1'].iloc[0]},
        {'model': 'LR + TF-IDF', 'dataset': 'FPB', 'macro_f1': lr_fpb['macro_f1']},
        {'model': 'EBM', 'dataset': 'FPB', 'macro_f1': ebm_fpb['macro_f1']},
        {'model': 'FinBERT+LR (Hybrid)', 'dataset': 'FPB', 'macro_f1': hybrid_fpb['macro avg']['f1-score']},

        {'model': 'VADER', 'dataset': 'FiQA', 'macro_f1': lexicon_df.query("model=='VADER' and dataset=='fiqa'")['macro_f1'].iloc[0]},
        {'model': 'LR + TF-IDF', 'dataset': 'FiQA', 'macro_f1': lr_fiqa['macro_f1']},
        {'model': 'EBM', 'dataset': 'FiQA', 'macro_f1': ebm_fiqa['macro_f1']},
        {'model': 'FinBERT+LR (Hybrid)', 'dataset': 'FiQA', 'macro_f1': hybrid_fiqa['macro avg']['f1-score']},

        {'model': 'VADER', 'dataset': 'Gold Standard', 'macro_f1': gold_df.query("model=='VADER'")['macro_f1'].iloc[0]},
        {'model': 'LR (Weak-Labeled)', 'dataset': 'Gold Standard', 'macro_f1': gold_df.query("model=='WEAKLR_OVERALL'")['macro_f1'].iloc[0]},
        {'model': 'FinBERT (Zero-Shot)', 'dataset': 'Gold Standard', 'macro_f1': gold_df.query("model=='FINBERT'")['macro_f1'].iloc[0]},
    ]
    return pd.DataFrame(data)

def main():
    FIGURE_PATH.parent.mkdir(exist_ok=True, parents=True)
    df = get_metrics()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(data=df, x='dataset', y='macro_f1', hue='model', ax=ax, palette='viridis')

    ax.set_title('Model Performance (Macro F1-Score) Across Datasets', fontsize=16, pad=20)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Macro F1-Score', fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8, padding=3)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.savefig(FIGURE_PATH, dpi=300)
    plt.close()

    print(f"✅ Performance summary chart saved to: {FIGURE_PATH}")

if __name__ == "__main__":
    main()