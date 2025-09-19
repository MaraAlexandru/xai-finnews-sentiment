# scripts/01_fetch_benchmarks.py
"""
Fetch FPB + FiQA (headlines only) → data/processed/*.csv and outputs/dataset_stats.csv

Zero-arg; uses Hugging Face `datasets`. If offline, prints a clear hint.
Run: python scripts/01_fetch_benchmarks.py
"""

from pathlib import Path
import pandas as pd

from _config import (
    FPB_CSV, FIQA_HEADLINES_CSV, OUTPUTS_DIR
)

def _save_stats(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "dataset": name,
        "n_rows": int(len(df)),
        "class_0_neg": int((df["y"] == 0).sum()) if "y" in df.columns else None,
        "class_1_neu": int((df["y"] == 1).sum()) if "y" in df.columns else None,
        "class_2_pos": int((df["y"] == 2).sum()) if "y" in df.columns else None,
    }])

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"[ERROR] `datasets` not available: {e}")
        print("Install with: pip install datasets\nRe-run this script with internet access.")
        return

    # FPB
    try:
        ds_fpb = load_dataset("takala/financial_phrasebank", "sentences_allagree")
        df_fpb = ds_fpb["train"].to_pandas()[["sentence","label"]].rename(columns={"sentence":"text","label":"y"})
        df_fpb.to_csv(FPB_CSV, index=False)
        print(f"Saved FPB → {FPB_CSV}")
    except Exception as e:
        print(f"[ERROR] FPB fetch failed: {e}")
        return

    # FiQA (headlines only, ±0.05 bin)
    try:
        ds_fiqa = load_dataset("TheFinAI/fiqa-sentiment-classification")
        frames = []
        for split in ds_fiqa.keys():
            d = ds_fiqa[split].to_pandas()[["sentence","score","type"]]
            d = d[d["type"] == "headline"].copy()
            d["y"] = pd.cut(d["score"], bins=[-1.01, -0.05, 0.05, 1.01], labels=[0,1,2]).astype(int)
            d = d.rename(columns={"sentence":"text"})[["text","y"]]
            frames.append(d)
        df_fiqa = pd.concat(frames, ignore_index=True)
        df_fiqa.to_csv(FIQA_HEADLINES_CSV, index=False)
        print(f"Saved FiQA (headlines) → {FIQA_HEADLINES_CSV}")
    except Exception as e:
        print(f"[ERROR] FiQA fetch failed: {e}")
        return

    # stats
    stats = pd.concat([
        _save_stats(df_fpb, "FPB"),
        _save_stats(df_fiqa, "FiQA_headlines")
    ], ignore_index=True)
    stats_out = OUTPUTS_DIR / "dataset_stats.csv"
    stats.to_csv(stats_out, index=False)
    print(f"Saved stats → {stats_out}")

if __name__ == "__main__":
    main()
