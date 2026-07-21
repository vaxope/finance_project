import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# Convert to long format where tickers are an actual column rather than baked into the column name
def load_prices_long(path: str, tickers: list[str]) -> pd.DataFrame:
    wide = pd.read_parquet(path)

    frames = []
    for t in tickers:
        sub = wide[t].copy()                              
        sub['ticker'] = t                                  
        sub = sub.reset_index().rename(columns={'Date': 'date', 'Close': 'close'})  
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)             

# Adds log returns to df
def add_log_returns(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=['ticker', 'date'])
    df['log_return'] = df.groupby('ticker')[price_col].transform(
        lambda x: np.log(x / x.shift(1))
    )
    return df


        