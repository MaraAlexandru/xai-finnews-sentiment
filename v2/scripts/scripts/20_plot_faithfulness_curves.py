# scripts/20_plot_faithfulness_curves.py
"""
Generates a line chart comparing the faithfulness of different XAI methods
for FinBERT, based on the deletion curve data.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "outputs" / "deep_xai" / "faithfulness_curves.csv"
FIGURE_PATH = ROOT / "outputs" / "figures" / "fig_faithfulness_curves.png"

def main():
    if not DATA_PATH.exists():
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        return

    FIGURE_PATH.parent.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(DATA_PATH)

    # Calculate mean probability at each removal fraction for each method
    curve_data = df.groupby(['method', 'frac_removed'])['prob'].mean().reset_index()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(
        data=curve_data,
        x='frac_removed',
        y='prob',
        hue='method',
        style='method',
        markers=True,
        dashes=False,
        ax=ax,
        linewidth=2.5
    )

    ax.set_title('Faithfulness of XAI Methods for FinBERT (Deletion Test)', fontsize=16, pad=20)
    ax.set_xlabel('Fraction of Top Tokens Removed', fontsize=12)
    ax.set_ylabel('Avg. Probability of Original Prediction', fontsize=12)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(bottom=0)
    ax.legend(title='XAI Method')
    
    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%'))
    
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=300)
    plt.close()

    print(f"✅ Faithfulness comparison chart saved to: {FIGURE_PATH}")


if __name__ == "__main__":
    main()