# scripts/21_plot_comparative_attributions.py
"""
Generates a comparative visualization of token attributions from IG, LIME,
and Attention Rollout for a single example.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import json
import numpy as np

# --- Configuration ---
ROOT = Path(__file__).resolve().parents[1]
XAI_DIR = ROOT / "outputs" / "deep_xai"
FIGURE_PATH = ROOT / "outputs" / "figures" / "fig_comparative_attributions.png"
EXAMPLE_INDEX = 23 # Pick an interesting example (e.g., one with strong neg/pos words)

# --- Helper to normalize and color text ---
def colorize_text(tokens, scores, ax, title):
    norm_scores = np.array(scores)
    # Scale scores to be mostly positive for color mapping
    norm_scores = (norm_scores - norm_scores.min()) / (norm_scores.max() - norm_scores.min() + 1e-9)

    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["#ffffff", "#ffeda0", "#f03b20"])
    
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.axis('off')

    words = []
    current_word = ""
    for token in tokens:
        if token in ['[CLS]', '[SEP]', '[PAD]']: continue
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word: words.append(current_word)
            current_word = token
    if current_word: words.append(current_word)
    
    # Re-align scores to words (simple averaging over subwords)
    word_scores = []
    score_idx = 0
    in_word = False
    temp_scores = []
    for token, score in zip(tokens, scores):
        if token in ['[CLS]', '[SEP]', '[PAD]']: continue
        if token.startswith("##"):
            temp_scores.append(score)
        else:
            if temp_scores:
                word_scores.append(np.mean(temp_scores))
            temp_scores = [score]
    if temp_scores:
         word_scores.append(np.mean(temp_scores))
    
    norm_word_scores = np.array(word_scores)
    norm_word_scores = (norm_word_scores - norm_word_scores.min()) / (norm_word_scores.max() - norm_word_scores.min() + 1e-9)


    # Use matplotlib's Text objects for layout
    text_obj = ax.text(0.01, 0.5, " ".join(words), ha='left', va='center', wrap=True, fontsize=10)
    
    # After drawing, get word positions and color them
    fig = plt.gcf()
    fig.canvas.draw()
    
    # This is a bit of a hack to get word positions, might need tweaking
    renderer = fig.canvas.get_renderer()
    words_on_canvas = text_obj.get_window_extent(renderer=renderer)
    
    x_pos = 0.01
    y_pos = 0.6
    
    for word, score in zip(words, norm_word_scores):
        t = ax.text(x_pos, y_pos, word, ha='left', va='center', fontsize=10,
                    bbox=dict(facecolor=cmap(score), alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'))
        
        # Advance x_pos, wrapping if necessary
        word_box = t.get_window_extent(renderer=renderer)
        fig_width = fig.get_window_extent().width
        x_pos += word_box.width / fig_width * 1.1 
        
        if x_pos > 0.95:
            x_pos = 0.01
            y_pos -= 0.2

    ax.set_title(title, loc='left', fontsize=12, pad=10)


def main():
    if not XAI_DIR.exists():
        print(f"❌ Error: XAI output directory not found at {XAI_DIR}")
        return

    # Load data for the chosen example
    try:
        with open(XAI_DIR / "ig" / "attributions.jsonl") as f:
            ig_data = [json.loads(line) for line in f][EXAMPLE_INDEX]
        with open(XAI_DIR / "lime" / "attributions.jsonl") as f:
            # LIME tokenization is different, so we'll just use its scores
            lime_data = [json.loads(line) for line in f][EXAMPLE_INDEX]
        with open(XAI_DIR / "attn" / "attributions.jsonl") as f:
            attn_data = [json.loads(line) for line in f][EXAMPLE_INDEX]
    except (FileNotFoundError, IndexError) as e:
        print(f"❌ Error loading attribution data: {e}. Check if files exist and example index is valid.")
        return

    # Use IG's tokenization as the reference
    tokens = ig_data['tokens']
    text = ig_data['text']
    
    # Align LIME scores to WordPiece tokens (crude but effective for viz)
    lime_scores_aligned = []
    lime_word_map = {word.lower(): score for word, score in zip(lime_data['tokens'], lime_data['attr'])}
    for token in tokens:
        clean_token = token.replace("##", "").lower()
        lime_scores_aligned.append(lime_word_map.get(clean_token, 0))


    fig, axes = plt.subplots(3, 1, figsize=(8, 4))
    plt.suptitle(f'Comparative Explanations for Headline:\n"{text[:100]}..."', fontsize=14, y=1.05)

    colorize_text(tokens, ig_data['attr'], axes[0], 'A) Integrated Gradients')
    colorize_text(tokens, lime_scores_aligned, axes[1], 'B) LIME')
    colorize_text(tokens, attn_data['attr'], axes[2], 'C) Attention Rollout')

    plt.tight_layout(pad=2.0)
    FIGURE_PATH.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Comparative attribution figure saved to: {FIGURE_PATH}")


if __name__ == "__main__":
    main()