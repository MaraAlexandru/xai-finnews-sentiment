# scripts/24_plot_shap_from_csv.py
"""
Generates SHAP summary bar charts directly from the pre-computed
'..._shap_top_tokens.csv' files. This is much faster and avoids
recalculating SHAP values.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# --- Configuration ---
ROOT = Path(__file__).resolve().parents[1]
WEAKLABEL_DIR = ROOT / "outputs" / "weaklabel_lr"
FIGURES_DIR = ROOT / "outputs" / "figures" / "shap_summaries_from_csv"
TOP_N_TOKENS = 20  # How many top tokens to display per plot

def generate_plot_from_csv(csv_path: Path, figure_path: Path, title: str):
    """
    Reads a pre-computed SHAP top tokens CSV and generates a bar plot.
    """
    if not csv_path.exists():
        print(f"⏩ Skipping '{title}': CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"⏩ Skipping '{title}': CSV file is empty.")
        return

    # Determine the top N tokens based on their average importance across all classes
    top_tokens = (
        df.groupby('token')['mean_abs_shap']
        .mean()
        .nlargest(TOP_N_TOKENS)
        .index
    )
    
    plot_data = df[df['token'].isin(top_tokens)].copy()
    
    # Map class integers to readable names
    class_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    plot_data['class_name'] = plot_data['class'].map(class_map)

    # Create the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.barplot(
        data=plot_data,
        y='token',
        x='mean_abs_shap',
        hue='class_name',
        order=top_tokens,  # Ensure consistent sorting
        palette={'Negative': '#d62728', 'Neutral': '#7f7f7f', 'Positive': '#2ca02c'},
        ax=ax
    )

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel('Token', fontsize=12)
    ax.set_xlabel('Mean Absolute SHAP Value (Impact on model output)', fontsize=12)
    ax.legend(title='Sentiment Class')

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"✅ SHAP summary plot for '{title}' saved to: {figure_path}")

def main():
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)

    # 1. Generate Overall Plot
    overall_csv = WEAKLABEL_DIR / "overall_shap_top_tokens.csv"
    generate_plot_from_csv(
        csv_path=overall_csv,
        figure_path=FIGURES_DIR / "fig_shap_summary_overall.png",
        title="Global SHAP Summary for All Sectors"
    )

    # 2. Generate Plot for each Industry
    industry_dir = WEAKLABEL_DIR / "by_industry"
    if industry_dir.exists():
        for csv_file in industry_dir.glob("*_shap_top_tokens.csv"):
            # Extract industry name from filename
            match = re.match(r"ind_(.*)_shap_top_tokens.csv", csv_file.name)
            if match:
                slug = match.group(1)
                industry_title = slug.replace('-', ' ').title()
                generate_plot_from_csv(
                    csv_path=csv_file,
                    figure_path=FIGURES_DIR / f"fig_shap_summary_{slug}.png",
                    title=f"SHAP Summary for {industry_title}"
                )

if __name__ == "__main__":
    main()