import yfinance as yf
import numpy as np
import pandas as pd
import requests
import io

# Pulls S&P 500 tickers from wikipedia table
def get_sp500_tickers():
    # User agent so we don' get blocked by wikipedia
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=headers
    )
    # Reads tickers and returns it as a list
    tables = pd.read_html(io.StringIO(res.text))
    df = tables[0]
    tickers = list(df.Symbol)
    return tickers

# Downloads ticker data
def fetch_prices(ticker, start="2023-01-01", end=None):
    data = yf.download(ticker, start=start, end=end, group_by="ticker",auto_adjust=True)
    return data

if __name__ == "__main__":
    tickers = get_sp500_tickers()
    df = fetch_prices(tickers)
    df.to_parquet("data/raw_prices.parquet")
    print(f"Saved {df.shape[0]} rows for {len(tickers)} tickers")

