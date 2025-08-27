# scripts/01_get_public_datasets.py
"""
Downloads and standardizes two public financial sentiment datasets:

1) Financial PhraseBank (FPB)
   - We use the 'sentences_allagree' variant (highest label agreement).
   - Hugging Face datasets v2 loader gives a split 'train' with columns:
       sentence (str), label (int in {0,1,2})
   - We rename to columns: text (str), y (int)

2) FiQA 2018 Sentiment Classification (mirror)
   - Contains 'sentence', 'score' (continuous), 'type' ('headline' or 'post').
   - We FILTER to type=='headline'.
   - We BIN the continuous score into 3 classes using thresholds ±0.05:
       y = 0 if score <= -0.05
       y = 1 if -0.05 < score < 0.05
       y = 2 if score >= 0.05
   - Output columns: text (str), y (int)

Outputs:
- data/processed/fpb.csv
- data/processed/fiqa_headlines.csv
- outputs/dataset_stats.csv (row counts & class distribution)

Why thresholds ±0.05?
- FiQA scores often cluster around 0 for neutral; ±0.05 is a common practical bin.
- You can tune later; keep it fixed for clean benchmarking.

NOTE: Requires `datasets` version 2.x (you already pinned) and internet access.
"""

from datasets import load_dataset
import pandas as pd
from pathlib import Path

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR = Path("outputs")
STATS_DIR.mkdir(parents=True, exist_ok=True)

def save_stats(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Create a one-row stats summary for a dataset."""
    return pd.DataFrame([{
        "dataset": name,
        "n_rows": len(df),
        "class_0_neg": int((df["y"] == 0).sum()),
        "class_1_neu": int((df["y"] == 1).sum()),
        "class_2_pos": int((df["y"] == 2).sum()),
    }])

def get_fpb() -> pd.DataFrame:
    """
    Load Financial PhraseBank (all-agree) and standardize.
    Labels are already {0:neg,1:neu,2:pos}.
    """
    ds = load_dataset("takala/financial_phrasebank", "sentences_allagree")
    df = ds["train"].to_pandas()[["sentence", "label"]].copy()
    df.rename(columns={"sentence": "text", "label": "y"}, inplace=True)
    df.to_csv(OUT_DIR / "fpb.csv", index=False)
    return df

def get_fiqa_headlines() -> pd.DataFrame:
    """
    Load FiQA sentiment classification (mirror), keep HEADLINES only,
    bin continuous scores into 3 classes with ±0.05 thresholds.
    """
    ds = load_dataset("TheFinAI/fiqa-sentiment-classification")
    def prep(split: str) -> pd.DataFrame:
        d = ds[split].to_pandas()[["sentence", "score", "type"]]
        d = d[d["type"] == "headline"].copy()
        # bin scores into 3 classes
        d["y"] = pd.cut(
            d["score"],
            bins=[-1.01, -0.05, 0.05, 1.01],
            labels=[0, 1, 2]
        ).astype(int)
        d.rename(columns={"sentence": "text"}, inplace=True)
        return d[["text", "y"]]
    df = pd.concat([prep(s) for s in ds.keys()], ignore_index=True)
    df.to_csv(OUT_DIR / "fiqa_headlines.csv", index=False)
    return df

if __name__ == "__main__":
    stats_rows = []
    fpb_df = get_fpb()
    fiqa_df = get_fiqa_headlines()

    stats_rows.append(save_stats(fpb_df, "FPB"))
    stats_rows.append(save_stats(fiqa_df, "FiQA_headlines"))

    stats = pd.concat(stats_rows, ignore_index=True)
    stats.to_csv(STATS_DIR / "dataset_stats.csv", index=False)
    print("Saved:")
    print(" - data/processed/fpb.csv")
    print(" - data/processed/fiqa_headlines.csv")
    print(" - outputs/dataset_stats.csv")
