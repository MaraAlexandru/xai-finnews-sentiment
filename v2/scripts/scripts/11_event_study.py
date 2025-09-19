# scripts/11_event_study.py
"""
Event Study: abnormal returns (AR/CAR) around high/low daily sentiment events.

If data/market/etf/SPY.csv exists → Market Model:
  r_sector_t = alpha + beta * r_spy_t + eps_t  (estimated in [-120,-21])
Else → Mean-adjusted model:
  AR_t = r_sector_t - mean(r_sector_t in [-120,-21])

Events: daily sentiment z-score (63d rolling) > +1.5 (POS) or < -1.5 (NEG).
Windows: estimation [-120,-21], event windows [-1,+3] and [-5,+5].

Inputs:
  data/processed/marketaux/marketaux_news_articles.csv
  outputs/weaklabel_lr/daily_sentiment_by_industry.csv  (optional; else auto-build)
  data/market/etf/{XLC,XLY,XLE,XLF,XLV,XLI,XLRE,XLU,XLK}.csv (sector ETFs)
  data/market/etf/SPY.csv (optional, preferred)

Outputs (per industry):
  outputs/event_study/<slug>/events.csv           (event dates & zscores)
  outputs/event_study/<slug>/car_table.csv        (CAR by window & side)
  outputs/event_study/<slug>/car_plot_<win>.png   (stacked CAR plot)
Global:
  outputs/event_study/summary.csv

Run: python scripts/11_event_study.py
"""
from pathlib import Path
import re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _config import MARKETAUX_ARTICLES_CSV, OUTPUTS_DIR, SEED

RNG = np.random.default_rng(SEED)

# --- Resolve project paths robustly (works even when run from /scripts) ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ETF_DIR = DATA_DIR / "market" / "etf"
OUT_DIR = OUTPUTS_DIR / "event_study"
SIG_CSV = OUTPUTS_DIR / "weaklabel_lr" / "daily_sentiment_by_industry.csv"
SPY_PATH = ETF_DIR / "SPY.csv"

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
    # "Services" → no clean SPDR proxy; skipped
}

def _slug(s):
    return re.sub(r"[^a-z0-9]+","-", (s or "unknown").lower()).strip("-")

def _read_etf(ticker: str):
    p = ETF_DIR / f"{ticker}.csv"
    if not p.exists():
        return None
    # Read with date parsing; drop any weird repeated-header rows automatically
    df = pd.read_csv(p, parse_dates=["Date"])
    # Drop rows where Date failed to parse or is missing (handles odd second header lines)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    # Make sure Adj Close is numeric
    df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
    df["date"] = df["Date"].dt.date
    df["ret"] = df["Adj Close"].pct_change()
    return df[["date","ret"]].dropna()

def _load_or_build_signal():
    if SIG_CSV.exists():
        d = pd.read_csv(SIG_CSV, parse_dates=["date"])
        d["date"] = d["date"].dt.date
        return d
    # Build from articles (EN only; weak labels via VADER→LM)
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
    # Weak labels
    y, src = _try_vader(df["text"].tolist())
    if y is None:
        y, src = _lm_label(df["text"].tolist())
    if y is None:
        print("[ERROR] Weak labels unavailable (need VADER or LM lexicon).")
        return None
    df["y"] = y
    df["s"] = df["y"].map({0:-1, 1:0, 2:+1})
    sig = (
        df.groupby(["date","dominant_industry"])["s"].mean()
          .reset_index()
          .rename(columns={"dominant_industry":"industry","s":"sent_mean"})
    )
    sig["industry"] = sig["industry"].fillna("Unknown")
    SIG_CSV.parent.mkdir(parents=True, exist_ok=True)
    sig.to_csv(SIG_CSV, index=False)
    print(f"[OK] Built daily sentiment → {SIG_CSV} (source={src})")
    return sig

def _try_vader(texts, pos_thr=0.05, neg_thr=-0.05):
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        import nltk
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon")
        sia = SentimentIntensityAnalyzer()
        labs = []
        for t in texts:
            c = sia.polarity_scores(t or "")["compound"]
            labs.append(2 if c >= pos_thr else 0 if c <= neg_thr else 1)
        return labs, "VADER"
    except Exception:
        return None, None

def _lm_label(texts):
    pos_p = DATA_DIR / "lexicons" / "lm" / "lm_positive.txt"
    neg_p = DATA_DIR / "lexicons" / "lm" / "lm_negative.txt"
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

def _zscore_roll(x, win=63, minp=40):
    s = x.rolling(win, min_periods=minp)
    return (x - s.mean()) / s.std(ddof=0)

def _event_windows():
    return {"[-1,+3]": (-1, 3), "[-5,+5]": (-5, 5)}

def _compute_ar_car(sec_ret, mkt_ret, event_date, est_win=(-120, -21), use_market_model=True):
    # Build estimation sample indices
    dates = sec_ret["date"].tolist()
    if event_date not in set(dates):
        return None
    idx = dates.index(event_date)
    est_start = idx + est_win[0]
    est_end   = idx + est_win[1]
    if est_start < 0 or est_end <= est_start:
        return None
    est = sec_ret.iloc[est_start:est_end+1].copy()
    if len(est) < 30:
        return None

    if use_market_model and mkt_ret is not None:
        est = est.merge(mkt_ret, on="date", how="inner", suffixes=("","_mkt")).dropna()
        if len(est) < 30:
            return None
        X = est["ret_mkt"].values
        Y = est["ret"].values
        X = np.vstack([np.ones_like(X), X]).T
        try:
            alpha, beta = np.linalg.lstsq(X, Y, rcond=None)[0]  # [alpha, beta]
        except Exception:
            return None
    else:
        alpha, beta = est["ret"].mean(), 0.0

    # Build CARs for each event window
    out = {}
    for name, (a, bwd) in _event_windows().items():
        lb = idx + a
        ub = idx + bwd
        if lb < 0 or ub >= len(sec_ret):
            out[name] = None
            continue
        window = sec_ret.iloc[lb:ub+1].copy()
        if mkt_ret is not None and use_market_model:
            window = window.merge(mkt_ret, on="date", how="inner", suffixes=("","_mkt")).dropna(subset=["ret","ret_mkt"])
            if window.empty:
                out[name] = None
                continue
            exp = alpha + beta * window["ret_mkt"].values
        else:
            exp = np.full(len(window), alpha)
        ar = window["ret"].values - exp
        car = np.cumsum(ar)
        out[name] = {
            "dates": window["date"].astype(str).tolist(),
            "ar": ar.tolist(),
            "car": car.tolist()
        }
    return out

def _tstat_mean(x):
    x = np.array(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return None
    m = x.mean()
    s = x.std(ddof=1)
    return None if s == 0 else float(m / (s / np.sqrt(len(x))))

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sig = _load_or_build_signal()
    if sig is None or sig.empty:
        return

    sig = sig.copy()
    sig["date"] = pd.to_datetime(sig["date"]).dt.date

    res = []
    spy = _read_etf("SPY") if SPY_PATH.exists() else None
    use_market_model = spy is not None

    for ind, ticker in SECTOR_TO_ETF.items():
        edir = OUT_DIR / _slug(ind)
        edir.mkdir(parents=True, exist_ok=True)

        sec = _read_etf(ticker)
        if sec is None:
            print(f"[WARN] Missing or unreadable ETF CSV for {ind} ({ticker}) at {ETF_DIR}.")
            continue

        s = sig[sig["industry"] == ind].dropna(subset=["sent_mean"]).copy()
        if s.empty:
            print(f"[INFO] No sentiment for {ind}.")
            continue

        # Align & z-score
        s = s.sort_values("date").reset_index(drop=True)
        s["z"] = _zscore_roll(
            pd.Series(s["sent_mean"].values, index=pd.to_datetime(s["date"])),
            win=63, minp=40
        ).values
        pos_ev = s[s["z"] >= 1.5]["date"].tolist()
        neg_ev = s[s["z"] <= -1.5]["date"].tolist()
        pd.DataFrame(
            {"side": ["POS"] * len(pos_ev) + ["NEG"] * len(neg_ev),
             "date": pos_ev + neg_ev}
        ).to_csv(edir / "events.csv", index=False)

        mkt = spy.copy() if use_market_model else None

        car_tables = []
        for side, ev_dates in {"POS": pos_ev, "NEG": neg_ev}.items():
            cars = {name: [] for name in _event_windows().keys()}
            for d in ev_dates:
                # FIXED: use the correct keyword 'use_market_model'
                arcar = _compute_ar_car(sec, mkt, d, use_market_model=use_market_model)
                if arcar is None:
                    continue
                for name, obj in arcar.items():
                    if obj is None or not obj["car"]:
                        continue
                    cars[name].append(obj["car"][-1])
            for name, vals in cars.items():
                vals = [v for v in vals if v is not None and np.isfinite(v)]
                n = len(vals)
                car_mean = float(np.mean(vals)) if n else None
                tstat = _tstat_mean(vals) if n else None
                car_tables.append({
                    "industry": ind, "ticker": ticker,
                    "model": "Market" if use_market_model else "MeanAdj",
                    "side": side, "window": name, "n_events": n,
                    "car_mean": car_mean, "tstat": tstat
                })

        car_df = pd.DataFrame(car_tables)
        car_df.to_csv(edir / "car_table.csv", index=False)

        # Simple bar plot for main window if any events
        try:
            main_win = "[-1,+3]"
            pos_vals = car_df[(car_df["side"] == "POS") & (car_df["window"] == main_win)]["car_mean"].dropna()
            neg_vals = car_df[(car_df["side"] == "NEG") & (car_df["window"] == main_win)]["car_mean"].dropna()
            if len(pos_vals) or len(neg_vals):
                fig = plt.figure(figsize=(5, 3.2))
                plt.bar(
                    ["POS", "NEG"],
                    [
                        pos_vals.mean() if len(pos_vals) else 0.0,
                        neg_vals.mean() if len(neg_vals) else 0.0
                    ]
                )
                plt.ylabel("Mean CAR")
                plt.title(f"{ind} — CAR {main_win}")
                plt.tight_layout()
                plt.savefig(edir / f"car_plot_{_slug(main_win)}.png", dpi=160)
                plt.close(fig)
        except Exception:
            pass

        res.extend(car_tables)

    if res:
        pd.DataFrame(res).to_csv(OUT_DIR / "summary.csv", index=False)
        print(f"[DONE] Event study summary → {OUT_DIR/'summary.csv'}")
    else:
        print(f"[WARN] No results written — check signals and ETF CSVs under: {ETF_DIR}")

if __name__ == "__main__":
    main()
