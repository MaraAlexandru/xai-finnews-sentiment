#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10_sentiment_to_returns.py
--------------------------

Purpose:
Connects news sentiment scores to subsequent stock market returns 

Workflow:
1.  Loads a CSV of articles with sentiment scores (e.g., from script 07).
2.  Parses tickers and dates from the news data.
3.  Aggregates sentiment scores to a daily level for each ticker.
4.  Proactively downloads historical stock price data for all relevant tickers
    using the yfinance library.
5.  Calculates next-day stock returns.
6.  Merges the daily sentiment data with the stock return data.
7.  Performs a correlation and simple linear regression analysis.
8.  Outputs the merged dataset, a summary of the analysis, and a scatter plot
    visualizing the relationship between sentiment and returns.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def run_analysis(args):
    """Main function to run the sentiment-return analysis pipeline."""
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    
    # --- 1. Load and Preprocess Sentiment Data ---
    print(f"Loading sentiment data from {args.sentiment_csv}...")
    try:
        sentiment_df = pd.read_csv(args.sentiment_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.sentiment_csv}")
        print("Please run script 07 or provide a CSV with 'published_at', 'tickers', and a sentiment score column.")
        return

    # Ensure required columns exist
    required_cols = ['published_at', 'tickers', args.sentiment_col]
    if not all(col in sentiment_df.columns for col in required_cols):
        raise ValueError(f"Input CSV must contain the columns: {required_cols}")

    # Clean and format dates and tickers
    sentiment_df['date'] = pd.to_datetime(sentiment_df['published_at'], utc=True, errors='coerce').dt.date
    sentiment_df.dropna(subset=['date', 'tickers', args.sentiment_col], inplace=True)
    
    # Explode the DataFrame to have one row per ticker for articles mentioning multiple tickers
    sentiment_df['tickers'] = sentiment_df['tickers'].str.split(',')
    exploded_df = sentiment_df.explode('tickers').reset_index(drop=True)
    exploded_df.rename(columns={'tickers': 'ticker'}, inplace=True)
    exploded_df['ticker'] = exploded_df['ticker'].str.strip()

    # --- 2. Aggregate Sentiment by Day and Ticker ---
    print("Aggregating sentiment scores by day and ticker...")
    daily_sentiment = exploded_df.groupby(['date', 'ticker'])[args.sentiment_col].mean().reset_index()
    daily_sentiment.rename(columns={args.sentiment_col: 'mean_sentiment'}, inplace=True)

    # --- 3. Fetch Historical Stock Prices ---
    unique_tickers = daily_sentiment['ticker'].unique().tolist()
    print(f"Found {len(unique_tickers)} unique tickers. Fetching historical price data...")
    
    price_data = yf.download(
        unique_tickers,
        start=args.start_date,
        end=args.end_date,
        progress=False # Set to True for verbose download progress
    )
    
    if price_data.empty:
        print("Could not download any price data for the given tickers and date range. Exiting.")
        return

    # Format the multi-index columns from yfinance
    price_data = price_data['Adj Close'].stack().reset_index()
    price_data.columns = ['date', 'ticker', 'adj_close']
    price_data['date'] = pd.to_datetime(price_data['date']).dt.date

    # --- 4. Calculate Next-Day Stock Returns ---
    print("Calculating next-day stock returns...")
    price_data.sort_values(by=['ticker', 'date'], inplace=True)
    price_data['next_day_return_pct'] = price_data.groupby('ticker')['adj_close'].pct_change().shift(-1) * 100
    price_data.dropna(subset=['next_day_return_pct'], inplace=True)

    # --- 5. Merge Sentiment and Returns Data ---
    print("Merging sentiment data with stock returns...")
    merged_df = pd.merge(daily_sentiment, price_data, on=['date', 'ticker'], how='inner')

    if merged_df.empty:
        print("Merge resulted in an empty DataFrame. Check for date/ticker alignment issues.")
        return
        
    # --- 6. Perform Analysis ---
    print("Performing correlation and regression analysis...")
    # Remove outliers for a cleaner analysis (e.g., returns > 50% in a day are rare)
    merged_df = merged_df[merged_df['next_day_return_pct'].abs() < 50]
    
    correlation = merged_df['mean_sentiment'].corr(merged_df['next_day_return_pct'])

    # Simple Linear Regression using statsmodels for more detail (p-value, R-squared)
    X = merged_df['mean_sentiment']
    y = merged_df['next_day_return_pct']
    X = sm.add_constant(X) # Adds a constant term to the predictor
    model = sm.OLS(y, X, missing='drop').fit()

    analysis_summary = (
        f"Sentiment vs. Next-Day Return Analysis\n"
        f"Period: {args.start_date} to {args.end_date}\n"
        f"Number of Observations (Ticker-Days): {len(merged_df)}\n"
        f"--------------------------------------------------\n"
        f"Pearson Correlation: {correlation:.4f}\n\n"
        f"Linear Regression Results:\n"
        f"R-squared: {model.rsquared:.4f}\n"
        f"Sentiment Coefficient: {model.params['mean_sentiment']:.4f}\n"
        f"P-value for Sentiment: {model.pvalues['mean_sentiment']:.4f}\n"
        f"--------------------------------------------------"
    )
    print("\n" + analysis_summary)

    # --- 7. Save Outputs ---
    # Save the merged data for further inspection
    merged_csv_path = out_dir / "sentiment_returns_merged.csv"
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"✅ Merged data saved to {merged_csv_path}")

    # Save the analysis summary
    summary_path = out_dir / "sentiment_returns_correlation.txt"
    with open(summary_path, 'w') as f:
        f.write(analysis_summary)
    print(f"✅ Analysis summary saved to {summary_path}")

    # Create and save the scatter plot
    plt.figure(figsize=(10, 6))
    sns.regplot(
        x='mean_sentiment',
        y='next_day_return_pct',
        data=merged_df,
        scatter_kws={'alpha': 0.3, 's': 20},
        line_kws={'color': 'red'}
    )
    plt.title(f'Sentiment Score vs. Next-Day Stock Return\n(Correlation: {correlation:.3f}, R-squared: {model.rsquared:.3f})', fontsize=16)
    plt.xlabel('Mean Daily Sentiment Score', fontsize=12)
    plt.ylabel('Next-Day Return (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = out_dir / "sentiment_vs_returns_scatter.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✅ Scatter plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link financial news sentiment to stock market returns.")
    parser.add_argument(
        "--sentiment_csv",
        type=str,
        required=True,
        help="Path to the CSV file containing sentiment-labeled articles. Must include 'published_at', 'tickers'."
    )
    parser.add_argument(
        "--sentiment_col",
        type=str,
        default="compound",
        help="Name of the column containing the numerical sentiment score (e.g., VADER's 'compound')."
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2025-01-01",
        help="Start date for the analysis in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2025-07-01",
        help="End date for the analysis in YYYY-MM-DD format (fetch a bit extra to calculate returns)."
    )
    args = parser.parse_args()
    run_analysis(args)