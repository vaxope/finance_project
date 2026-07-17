import yfinance as yf
import numpy as np
import pandas as pd

tickers = ['SPY', 'VOO', 'AAPL', 'MSFT', 'NVDA', 'QQQ']

def fetch_prices(ticker, start="2015-01-01", end=None):
    data = yf.download(tickers, start=start, end=end, group_by="ticker",auto_adjust=True)
    return data

if __name__ == "__main__":
    df = fetch_prices(tickers)
    df.to_parquet("data/raw_prices.parquet")
    print(f"Saved {df.shape[0]} rows for {len(tickers)} tickers")