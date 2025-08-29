#xai-finnews-sentiment

Explainable financial news sentiment toolkit. Reproducible pipelines for:

Interpretable TF-IDF + Logistic Regression (with SHAP)

FinBERT inference & fine-tuning

Loughran–McDonald (LM) lexicon baselines & VADER

Marketaux news preparation, weak-supervision, and regime-shift token analysis

All major figures, tables, and reports in outputs/ are produced by scripts in scripts/.

Folder layout
data/
  annotation/            # manual gold labels + sampling sheet
  lexicons/lm/           # LM lists (positive, negative, etc.)
  processed/             # standardized CSVs used by scripts
  raw/marketaux/         # your Marketaux export (news.json)
models/                  # FinBERT checkpoints (if fine-tuned)
outputs/                 # metrics, confusion matrices, SHAP, figs
scripts/                 # all CLI pipelines (see below)
Loughran-McDonald_MasterDictionary_1993-2024.xlsx
README.txt

Installation

Python 3.10+ recommended.

pip install -U pandas numpy scikit-learn nltk shap matplotlib datasets transformers accelerate "torch==2.*" sentencepiece yfinance statsmodels
# (optional, if NLTK prompts for it)
python -m nltk.downloader vader_lexicon


If you plan to use a GPU with PyTorch, install a CUDA-matched torch wheel from PyTorch docs.

Quickstart
1) Get public datasets (FPB, FiQA headlines)
python scripts/01_get_public_datasets.py


Creates:

data/processed/fpb.csv

data/processed/fiqa_headlines.csv (FiQA scores binned to 3 classes using ±0.05 thresholds)

2) Convert your Marketaux export

Place your export at data/raw/marketaux/news.json (example format shown in repo), then:

python scripts/01b_marketaux_json_to_csv.py \
  --json data/raw/marketaux/news.json \
  --outdir data/processed


Creates marketaux_news_articles.csv, marketaux_news_pairs.csv, and an annotation sheet if requested.

3) Interpretable baseline (TF-IDF + LR with SHAP)
python scripts/02_run_lr_shap.py --csv data/processed/fpb.csv
python scripts/02_run_lr_shap.py --csv data/processed/fiqa_headlines.csv


Outputs: lr_metrics_*.csv, lr_classification_report_*.txt, shap_summary_*.png, top_tokens_shap_*.csv.

4) FinBERT (zero-shot or fine-tune)
# zero-shot evaluation
python scripts/03_run_finbert.py --csv data/processed/fpb.csv --mode inference
python scripts/03_run_finbert.py --csv data/processed/fiqa_headlines.csv --mode inference

# optional fine-tuning (saves to models/)
python scripts/03_run_finbert.py --csv data/processed/fiqa_headlines.csv --mode finetune


Downloads ProsusAI/finbert from Hugging Face on first run.

5) LM lexicon ablations (+ optional VADER compare)

LM lists are in data/lexicons/lm/. To regenerate them from the official Excel:

python scripts/04a_build_lm_from_xlsx.py


Run ablations:

python scripts/04_lexicon_ablation.py --csv data/processed/fpb.csv --compare_vader 1
python scripts/04_lexicon_ablation.py --csv data/processed/fiqa_headlines.csv --compare_vader 1

6) Prepare annotation sample & evaluate gold labels
# stratified sampling sheet (optional)
python scripts/05_prepare_marketaux.py \
  --json data/raw/marketaux/news.json \
  --outdir data/processed --sample_out data/annotation --sample_size 100

# evaluate VADER, LR, FinBERT on manual labels
python scripts/06_eval_manual_labels.py \
  --annot_csv data/annotation/annotated_articles.csv \
  --train_csvs data/processed/fpb.csv,data/processed/fiqa_headlines.csv


Creates 06_* reports, confusion matrices, and combined predictions.

7) Weak supervision & regime shift analysis
# SHAP on filtered Marketaux slice
python scripts/07_analyze_marketaux_revised.py \
  --csv data/processed/marketaux_news_articles.csv \
  --include_industries "semiconductors,technology" \
  --start_date 2025-01-01 --end_date 2025-06-30

# Q1 vs Q2 token rank-shift
python scripts/08_shap_regime_shift.py \
  --csv data/processed/marketaux_news_articles.csv \
  --year 2025 --include_industries "technology"

8) Paper figure assembly (confusion matrices panel)
python scripts/09_plot_confusion_matrices.py


Saves outputs/Figure1_Confusion_Matrices.png.

Key outputs (examples)

*classification_report*.txt, *metrics*.csv — model metrics

*confmat*.csv and Figure1_Confusion_Matrices.png — confusion matrices

shap_summary_*.png, top_tokens_shap_*.csv — explainability artifacts

regime_tokens_*.csv, regime_rank_shift_*.csv, regime_shift_top20_*.png — temporal/drivers analysis

Data sources & downloads

Financial PhraseBank and FiQA 2018 are downloaded from Hugging Face by 01_get_public_datasets.py.

ProsusAI/finbert is downloaded by 03_run_finbert.py (first run).

VADER lexicon is fetched by NLTK if missing.

LM lexicons are provided in data/lexicons/lm/ and can be regenerated locally from Loughran-McDonald_MasterDictionary_1993-2024.xlsx via 04a_build_lm_from_xlsx.py.

Marketaux news are not fetched by the repo; place your API export at data/raw/marketaux/news.json.

Licensing

Code (this repo): MIT License

Manual labels (data/annotation/annotated_articles.csv) and docs: CC BY 4.0

Third-party assets: original licenses/terms apply

Loughran–McDonald Master Dictionary: downloaded from Notre Dame SRAF; check their terms before redistribution.

Marketaux content: subject to Marketaux API Terms.

ProsusAI/finbert, FPB, FiQA, VADER: governed by their original licenses.

Reproducibility notes

Random seeds are set in scripts where applicable; minor variance may occur across library versions.

CPU-only works for everything except FinBERT fine-tuning (GPU recommended).

Citation

If this repo helps your work, please cite:

@software{XAI_FinNews_Sentiment_2025,
  title  = {xai-finnews-sentiment: Explainable Financial News Sentiment Toolkit},
  year   = {2025},
  author = {Repo Authors},
  url    = {https://github.com/<your-org>/xai-finnews-sentiment}
}

Acknowledgements

Loughran & McDonald for the Master Dictionary

ProsusAI for FinBERT

FPB and FiQA dataset contributors

Marketaux for API access
