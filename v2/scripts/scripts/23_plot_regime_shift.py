# scripts/23_plot_regime_shift.py
"""
Generates an updated temporal regime shift plot for ALL industries that have
the required output files from the regime analysis.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
ROOT = Path(__file__).resolve().parents[1]
REGIME_DIR = ROOT / "outputs" / "regime"
FIGURES_DIR = ROOT / "outputs" / "figures" / "regime_shifts"
TOP_N_SHIFT = 10 # Number of top gainers and top losers to show

def generate_plot(industry_slug: str, figure_path: Path):
    """Helper function to generate a single regime shift plot."""
    early_path = REGIME_DIR / industry_slug / "time_top_tokens_early.csv"
    late_path = REGIME_DIR / industry_slug / "time_top_tokens_late.csv"

    if not early_path.exists() or not late_path.exists():
        print(f"⏩ Skipping '{industry_slug}': Missing early or late token files.")
        return

    df_early = pd.read_csv(early_path)
    df_late = pd.read_csv(late_path)
    
    if df_early.empty or df_late.empty:
        print(f"⏩ Skipping '{industry_slug}': One of the token files is empty.")
        return

    # Aggregate rank across classes to get an overall importance score
    early_ranks = df_early.groupby('token')['rank'].mean().reset_index().rename(columns={'rank': 'rank_early'})
    late_ranks = df_late.groupby('token')['rank'].mean().reset_index().rename(columns={'rank': 'rank_late'})

    # Merge and calculate shift
    df_merged = pd.merge(early_ranks, late_ranks, on='token', how='outer')
    max_rank = max(df_merged['rank_early'].max(), df_merged['rank_late'].max()) + 1
    df_merged = df_merged.fillna(max_rank)
    
    df_merged['rank_shift'] = df_merged['rank_early'] - df_merged['rank_late']
    df_merged = df_merged.sort_values('rank_shift', ascending=False)

    # Get top gainers (largest positive shift) and top losers (largest negative shift)
    top_gainers = df_merged.head(TOP_N_SHIFT)
    top_losers = df_merged.tail(TOP_N_SHIFT).sort_values('rank_shift', ascending=False)
    plot_data = pd.concat([top_gainers, top_losers])
    
    # Create the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#2ca02c' if x > 0 else '#d62728' for x in plot_data['rank_shift']]
    sns.barplot(data=plot_data, x='rank_shift', y='token', palette=colors, ax=ax)

    industry_title = industry_slug.replace('-', ' ').title()
    ax.set_title(f'Top Tokens by Change in Importance for {industry_title}\n(Early vs. Late Period)', fontsize=16, pad=20)
    ax.set_xlabel('Rank Shift (Positive value means gained importance)', fontsize=12)
    ax.set_ylabel('Token', fontsize=12)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    
    # Add rank annotations
    for i, (index, row) in enumerate(plot_data.iterrows()):
        text = f"Rank: {int(row['rank_early'])} → {int(row['rank_late'])}"
        x_pos = row['rank_shift']
        offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02 # 2% of plot width
        ha = 'left' if x_pos >= 0 else 'right'
        ax.text(x_pos + (offset if ha == 'left' else -offset), i, text, va='center', ha=ha, fontsize=9)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"✅ Regime shift plot saved for '{industry_slug}' to: {figure_path}")

def main():
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    
    if not REGIME_DIR.exists():
        print(f"❌ Error: Regime directory not found at {REGIME_DIR}.")
        print("Please run '10_shap_regime_shift_all.py' first.")
        return

    # Find all industry subdirectories that contain the necessary files
    industry_slugs = [d.name for d in REGIME_DIR.iterdir() if d.is_dir()]
    
    for slug in industry_slugs:
        figure_path = FIGURES_DIR / f"fig_regime_shift_{slug}.png"
        generate_plot(slug, figure_path)

if __name__ == "__main__":
    main()