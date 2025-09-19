# scripts/12_granger_var.py
"""
Granger/VAR tests: does daily sentiment Granger-cause sector ETF returns?

Procedure per industry with ETF:
  - Build/Load daily sentiment mean (EN, weak labels).
  - Align with ETF daily returns.
  - Standardize both series.
  - Fit VAR, lag order by AIC (adaptive cap).
  - Test causality:
      (a) sentiment → returns
      (b) returns → sentiment

Outputs:
  outputs/granger/<slug>/var_summary.json
  outputs/granger/<slug>/residuals.csv
Global:
  outputs/granger/summary.csv  (p-values & chosen lags)

Run: python scripts/12_granger_var.py
"""
from pathlib import Path
import re, json
import numpy as np
import pandas as pd

from _config import MARKETAUX_ARTICLES_CSV, OUTPUTS_DIR, SEED

# ---------------- Paths (robust even when run from /scripts) ----------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ETF_DIR = DATA_DIR / "market" / "etf"
LEX_DIR = DATA_DIR / "lexicons" / "lm"
OUT_DIR = OUTPUTS_DIR / "granger"
SIG_CSV = OUTPUTS_DIR / "weaklabel_lr" / "daily_sentiment_by_industry.csv"

# ---------------- Settings ----------------
RNG = np.random.default_rng(SEED)
MIN_OBS = 25          # minimum aligned observations to attempt estimation
MAX_LAG_CAP = 5       # global cap; adaptive will further reduce it
NEAR_CONST_STD = 1e-12  # treat as constant if std < this

SECTOR_TO_ETF = {
    "Communication Services": "XLC",
    "Consumer Cyclical":      "XLY",
    "Energy":                 "XLE",
    "Financial Services":     "XLF",
    "Healthcare":             "XLV",
    "Industrials":            "XLI",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
    "Technology":             "XLK",
}

# ---------------- Helpers ----------------
def _slug(s): 
    return re.sub(r"[^a-z0-9]+","-", (s or "unknown").lower()).strip("-")

def _read_etf(ticker: str):
    p = ETF_DIR / f"{ticker}.csv"
    if not p.exists():
        return None
    # Robust read; some downloads add a second header line
    df = pd.read_csv(p)
    # Drop obvious second-header rows like ",XLC,XLC,..."
    if df.shape[0] and (pd.isna(df.iloc[0, 0]) or str(df.iloc[0, 0]).strip() == ""):
        df = df.iloc[1:].reset_index(drop=True)
    if "Date" not in df.columns or "Adj Close" not in df.columns:
        df = pd.read_csv(p, parse_dates=["Date"])
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["date"] = df["Date"].dt.date
    df["ret"] = pd.to_numeric(df["Adj Close"], errors="coerce").pct_change()
    return df[["date","ret"]].dropna()

def _try_vader(texts, pos_thr=0.05, neg_thr=-0.05):
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        import nltk
        try: nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError: nltk.download("vader_lexicon")
        sia = SentimentIntensityAnalyzer()
        labs = []
        for t in texts:
            c = sia.polarity_scores(t or "")["compound"]
            labs.append(2 if c >= pos_thr else 0 if c <= neg_thr else 1)
        return labs, "VADER"
    except Exception:
        return None, None

def _lm_label(texts):
    pos_p = LEX_DIR / "lm_positive.txt"
    neg_p = LEX_DIR / "lm_negative.txt"
    if not (pos_p.exists() and neg_p.exists()):
        return None, None
    pos = set(w.strip().lower() for w in pos_p.read_text(encoding="utf-8").splitlines() if w.strip())
    neg = set(w.strip().lower() for w in neg_p.read_text(encoding="utf-8").splitlines() if w.strip())
    import re
    tok_re = re.compile(r"[A-Za-z]+")
    y = []
    for t in texts:
        toks = [x.lower() for x in tok_re.findall(t or "")]
        p = sum((z in pos) for z in toks); n = sum((z in neg) for z in toks)
        sc = p - n
        y.append(2 if sc >= 1 else 0 if sc <= -1 else 1)
    return y, "LM"

def _load_or_build_signal():
    if SIG_CSV.exists():
        d = pd.read_csv(SIG_CSV, parse_dates=["date"])
        d["date"] = d["date"].dt.date
        return d
    # Build EN weak-label signal if missing
    if not MARKETAUX_ARTICLES_CSV.exists():
        print(f"[ERROR] Missing: {MARKETAUX_ARTICLES_CSV}")
        return None
    df = pd.read_csv(MARKETAUX_ARTICLES_CSV, parse_dates=["published_at"])
    df["date"] = df["published_at"].dt.date
    df = df[df["language"].fillna("unk").str.lower().eq("en")].copy()
    df["text"] = df["text"].fillna("")
    if df.empty:
        print("[ERROR] No EN articles to build signal.")
        return None
    y, src = _try_vader(df["text"].tolist())
    if y is None:
        y, src = _lm_label(df["text"].tolist())
    if y is None:
        print("[ERROR] Weak labels unavailable.")
        return None
    df["y"] = y
    df["s"] = df["y"].map({0:-1,1:0,2:+1})
    sig = (df.groupby(["date","dominant_industry"])["s"].mean()
             .reset_index().rename(columns={"dominant_industry":"industry","s":"sent_mean"}))
    sig["industry"] = sig["industry"].fillna("Unknown")
    SIG_CSV.parent.mkdir(parents=True, exist_ok=True)
    sig.to_csv(SIG_CSV, index=False)
    print(f"[OK] Built signal → {SIG_CSV}")
    return sig

def _standardize(a: pd.Series):
    a = pd.to_numeric(a, errors="coerce")
    mu = a.mean()
    sd = a.std(ddof=0)
    if not np.isfinite(sd) or sd < NEAR_CONST_STD:
        return None  # flag near-constant series
    return (a - mu) / sd

# ---------------- Main ----------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import statsmodels.api as sm  # noqa
        from statsmodels.tsa.api import VAR
        from statsmodels.tsa.stattools import grangercausalitytests
    except Exception as e:
        print(f"[ERROR] statsmodels required: {e}")
        return

    sig = _load_or_build_signal()
    if sig is None or sig.empty:
        return

    rows = []
    for ind, ticker in SECTOR_TO_ETF.items():
        edir = OUT_DIR / _slug(ind)
        edir.mkdir(parents=True, exist_ok=True)

        sec = _read_etf(ticker)
        if sec is None:
            print(f"[WARN] Missing or unreadable ETF for {ind} ({ticker}) at {ETF_DIR}.")
            continue

        s = sig[sig["industry"]==ind][["date","sent_mean"]].dropna()
        if s.empty:
            print(f"[INFO] No sentiment for {ind}.")
            continue

        # Align and clean
        dfm = sec.merge(s, on="date", how="inner").dropna()
        if len(dfm) < MIN_OBS:
            print(f"[INFO] Not enough data for VAR ({ind}). Need ≥{MIN_OBS}, got {len(dfm)}.")
            continue

        dfm = dfm.sort_values("date").reset_index(drop=True)

        # Standardize; skip if a series is (near) constant
        z_ret  = _standardize(dfm["ret"])
        z_sent = _standardize(dfm["sent_mean"])
        if z_ret is None or z_sent is None:
            print(f"[INFO] Skipping {ind}: series near-constant after alignment (std too small).")
            continue

        X = pd.DataFrame({"ret": z_ret.astype("float64"),
                          "sent": z_sent.astype("float64")})
        # Drop any residual non-finite values
        X = X.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        if len(X) < MIN_OBS:
            print(f"[INFO] Not enough clean rows for VAR ({ind}) after filtering.")
            continue

        # Use a simple RangeIndex to avoid date frequency warnings
        X.index = pd.RangeIndex(start=0, stop=len(X), step=1)

        # Adaptive max lag: ~15 obs per lag, capped
        auto_maxlag = max(1, min(MAX_LAG_CAP, len(X)//15))

        # Choose lag by AIC (guardrails)
        try:
            sel = VAR(X).select_order(auto_maxlag)
            aic_lag = getattr(sel, "aic", None)
            p = int(aic_lag) if aic_lag is not None else 2
            p = max(1, min(auto_maxlag, p))
            # also ensure enough df: rough guard (nobs > 8*p)
            while p > 1 and len(X) <= 8*p:
                p -= 1
        except Exception:
            p = max(1, min(auto_maxlag, 2))

        # Fit VAR with backoff if needed
        model = None
        for trial_p in range(p, 0, -1):
            try:
                model = VAR(X).fit(trial_p)
                p = trial_p
                break
            except Exception:
                continue

        if model is None:
            # Fallback: classical Granger on pairs; take min p across lags
            try:
                g1 = grangercausalitytests(X[["ret","sent"]].values, maxlag=auto_maxlag, verbose=False)
                p_sent_to_ret = float(min(v[0]["ssr_ftest"][1] for _, v in g1.items()))
                g2 = grangercausalitytests(X[["sent","ret"]].values, maxlag=auto_maxlag, verbose=False)
                p_ret_to_sent = float(min(v[0]["ssr_ftest"][1] for _, v in g2.items()))
                summ = {
                    "industry": ind, "ticker": ticker, "lag_order": None,
                    "n_obs": int(len(X)),
                    "p_sent_to_ret": p_sent_to_ret,
                    "p_ret_to_sent": p_ret_to_sent,
                    "fallback": "grangercausalitytests"
                }
                (edir / "var_summary.json").write_text(json.dumps(summ, indent=2))
                rows.append({
                    "industry": ind, "ticker": ticker, "lag_order": None, "n_obs": int(len(X)),
                    "p_sent_to_ret": p_sent_to_ret, "p_ret_to_sent": p_ret_to_sent
                })
                continue
            except Exception as e:
                print(f"[WARN] Both VAR and fallback Granger failed for {ind}: {e}")
                continue

        # VAR causality tests
        try:
            c1 = model.test_causality("ret", ["sent"], kind="f")
            p_sent_to_ret = float(c1.pvalue)
        except Exception:
            p_sent_to_ret = None
        try:
            c2 = model.test_causality("sent", ["ret"], kind="f")
            p_ret_to_sent = float(c2.pvalue)
        except Exception:
            p_ret_to_sent = None

        # Save compact summary (params castable to float JSON)
        try:
            params_dict = model.params.astype(float).to_dict()
        except Exception:
            params_dict = {str(k): float(v) for k, v in np.ndenumerate(model.params.values)}

        summ = {
            "industry": ind, "ticker": ticker, "lag_order": p,
            "n_obs": int(model.nobs),
            "p_sent_to_ret": p_sent_to_ret,
            "p_ret_to_sent": p_ret_to_sent,
            "params": params_dict,
        }
        (edir / "var_summary.json").write_text(json.dumps(summ, indent=2))

        # residuals (optional)
        try:
            model.resid.to_csv(edir / "residuals.csv")
        except Exception:
            pass

        rows.append({
            "industry": ind, "ticker": ticker, "lag_order": p, "n_obs": int(model.nobs),
            "p_sent_to_ret": p_sent_to_ret, "p_ret_to_sent": p_ret_to_sent
        })

    if rows:
        pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False)
        print(f"[DONE] Granger/VAR summary → {OUT_DIR/'summary.csv'}")
    else:
        print("[WARN] No VAR results written.")

if __name__ == "__main__":
    main()
