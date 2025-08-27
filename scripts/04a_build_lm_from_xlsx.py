#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
04a_build_lm_from_xlsx.py
-------------------------

Convert the official Loughran–McDonald Master Dictionary (XLSX)
into seven plain-text category lists used by our lexicon ablation:

  positive.txt
  negative.txt
  uncertainty.txt
  litigious.txt
  modal_strong.txt
  modal_weak.txt
  constraining.txt

Assumptions:
- The Master Dictionary has columns like:
  Word, Negative, Positive, Uncertainty, Litigious, Strong_Modal (or "Strong Modal"),
  Weak_Modal (or "Weak Modal"), Constraining
- Category columns are either 0 or a year (non-zero => include word)

Usage (single line; quotes required because of spaces in the path):
  python scripts/04a_build_lm_from_xlsx.py --xlsx "C:\Users\Mara\Digi Storage folder\xai_v2\Loughran-McDonald_MasterDictionary_1993-2024.xlsx" --out "C:\Users\Mara\Digi Storage folder\xai_v2\data\lexicons\lm"

Outputs:
  <out>\positive.txt, <out>\negative.txt, ... (one lowercase token per line)
  <out>\README.txt   (counts and notes)
"""

import argparse
from pathlib import Path
import re
import sys

import pandas as pd

# Accept multiple column name variants (case-insensitive, spaces/underscores)
COL_VARIANTS = {
    "word": ["word"],
    "negative": ["negative"],
    "positive": ["positive"],
    "uncertainty": ["uncertainty"],
    "litigious": ["litigious"],
    "strong_modal": ["strong_modal", "strong modal", "modal_strong", "strongmodal"],
    "weak_modal": ["weak_modal", "weak modal", "modal_weak", "weakmodal"],
    "constraining": ["constraining"],
}

SAFE_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-\']+$")

def normalize_colname(s: str) -> str:
    s = str(s)
    s = s.strip().lower().replace("  ", " ")
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("_", " ")
    return s

def find_column(df: pd.DataFrame, logical_name: str) -> str:
    """Find the real column name in df that matches the logical category."""
    candidates = COL_VARIANTS[logical_name]
    norm_map = {normalize_colname(c): c for c in df.columns}
    # direct matches for known variants
    for cand in candidates:
        n = normalize_colname(cand)
        if n in norm_map:
            return norm_map[n]
    # fuzzy fallback: look for the logical name words within a normalized header
    needle = logical_name.replace("_", " ")
    for col in df.columns:
        if needle in normalize_colname(col):
            return col
    raise KeyError(
        f"Could not find a column for '{logical_name}'. "
        f"Available columns: {list(df.columns)}"
    )

def clean_token(w: str) -> str:
    w = str(w).strip()
    if not w:
        return ""
    w = w.split()[0]              # drop trailing annotations if any
    w = w.replace("’","'").replace("`","'")
    w = w.replace("–","-").replace("—","-")
    w = w.lower()
    return w

def is_valid_token(w: str) -> bool:
    # keep words that start with a letter (+ allow digits, dash, apostrophe later)
    return bool(SAFE_TOKEN_RE.match(w))

def load_first_sheet(xlsx_path: str, sheet: str | None):
    """
    Robust Excel loader:
      - if sheet is provided: load that sheet
      - else: load the FIRST sheet (sheet_name=0)
    Avoids returning a dict (sheet_name=None returns all sheets).
    """
    try:
        sheet_arg = 0 if sheet is None else sheet
        df = pd.read_excel(xlsx_path, sheet_name=sheet_arg, engine="openpyxl")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read Excel: {e}", file=sys.stderr)
        raise

def main(xlsx_path: str, out_dir: str, sheet: str | None = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Read Excel (first sheet by default, NOT all sheets)
    df = load_first_sheet(xlsx_path, sheet)

    # Ensure we have a DataFrame, not a dict
    if isinstance(df, dict):
        # If pandas still returned a dict for some reason, pick the first sheet
        first_key = next(iter(df))
        df = df[first_key]

    # Locate columns (your headers like 'Word', 'Negative', 'Strong_Modal' will be matched)
    col_word         = find_column(df, "word")
    col_negative     = find_column(df, "negative")
    col_positive     = find_column(df, "positive")
    col_uncertainty  = find_column(df, "uncertainty")
    col_litigious    = find_column(df, "litigious")
    col_strong_modal = find_column(df, "strong_modal")
    col_weak_modal   = find_column(df, "weak_modal")
    col_constraining = find_column(df, "constraining")

    print("[INFO] Column mapping:")
    print(f"  Word          -> {col_word}")
    print(f"  Negative      -> {col_negative}")
    print(f"  Positive      -> {col_positive}")
    print(f"  Uncertainty   -> {col_uncertainty}")
    print(f"  Litigious     -> {col_litigious}")
    print(f"  Strong_Modal  -> {col_strong_modal}")
    print(f"  Weak_Modal    -> {col_weak_modal}")
    print(f"  Constraining  -> {col_constraining}")

    # Any non-zero / non-empty entry => include in the list
    def extract(cat_col: str) -> list[str]:
        words = []
        for _, row in df[[col_word, cat_col]].iterrows():
            w = clean_token(row[col_word])
            if not w or not is_valid_token(w):
                continue
            val = row[cat_col]
            try:
                nz = (float(val) != 0.0)
            except Exception:
                nz = (str(val).strip() not in {"", "0", "0.0", "nan", "NaN"})
            if nz:
                words.append(w)
        # de-duplicate preserving order
        seen, uniq = set(), []
        for w in words:
            if w not in seen:
                seen.add(w); uniq.append(w)
        return uniq

    lists = {
        "positive.txt":     extract(col_positive),
        "negative.txt":     extract(col_negative),
        "uncertainty.txt":  extract(col_uncertainty),
        "litigious.txt":    extract(col_litigious),
        "modal_strong.txt": extract(col_strong_modal),
        "modal_weak.txt":   extract(col_weak_modal),
        "constraining.txt": extract(col_constraining),
    }

    # Write files
    for fname, words in lists.items():
        (out / fname).write_text("\n".join(words) + "\n", encoding="utf-8")
    # Write readme with counts
    lines = [f"{k}: {len(v)} terms" for k, v in lists.items()]
    (out / "README.txt").write_text(
        "Loughran–McDonald category lists generated from Master Dictionary XLSX\n\n" +
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("✓ Wrote LM lists to", out.resolve())
    for fname, words in lists.items():
        print(f"  {fname:18s} {len(words):6d} terms")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Path to Loughran–McDonald Master Dictionary .xlsx")
    ap.add_argument("--out",  required=True, help="Output directory for the seven LM text lists")
    ap.add_argument("--sheet", default=None, help="Optional sheet name if you want a specific sheet")
    args = ap.parse_args()
    main(args.xlsx, args.out, args.sheet)
