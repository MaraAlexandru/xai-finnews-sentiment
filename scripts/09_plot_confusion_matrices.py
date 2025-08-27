# scripts/09_plot_confusion_matrices.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_confusion_matrices():
    """
    Reads the three confusion matrix CSVs from script 06's output
    and generates a single 1x3 plot for Figure 1 in the paper.
    """
    output_dir = Path("outputs")
    
    try:
        # Load the data from your output files
        cm_vader = pd.read_csv(output_dir / "06_confmat_annotated_articles_VADER.csv", index_col=0)
        cm_lr = pd.read_csv(output_dir / "06_confmat_annotated_articles_LR.csv", index_col=0)
        cm_finbert = pd.read_csv(output_dir / "06_confmat_annotated_articles_FinBERT.csv", index_col=0)
    except FileNotFoundError as e:
        print(f"Error: Could not find a confusion matrix file. {e}")
        print("Please ensure you have run script 06_eval_manual_labels.py first.")
        return

    # --- Plotting ---
    # Create a figure with 3 subplots side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle('Model Performance on Manually Labeled Headlines (N=100)', fontsize=18, y=1.03)
    
    labels = ['Negative', 'Neutral', 'Positive']
    
    # Plot 1: VADER
    sns.heatmap(cm_vader, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"size": 14})
    axes[0].set_title('VADER (Rule-Based)', fontsize=14)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)

    # Plot 2: LR + TF-IDF
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"size": 14})
    axes[1].set_title('LR + TF-IDF (Interpretable)', fontsize=14)
    axes[1].set_xlabel('Predicted Label', fontsize=12)

    # Plot 3: FinBERT
    sns.heatmap(cm_finbert, annot=True, fmt='d', cmap='Oranges', ax=axes[2],
                xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"size": 14})
    axes[2].set_title('FinBERT (Zero-Shot)', fontsize=14)
    axes[2].set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the final figure
    save_path = output_dir / "Figure1_Confusion_Matrices.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Figure 1 has been successfully generated and saved to: {save_path}")

if __name__ == "__main__":
    plot_confusion_matrices()