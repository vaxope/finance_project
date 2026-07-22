import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import talib

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# Convert to long format where tickers are an actual column rather than baked into the column name
def load_prices_long(path: str, tickers: list[str]) -> pd.DataFrame:
    wide = pd.read_parquet(path)

    frames = []
    for t in tickers:
        sub = wide[t].copy()                              
        sub['ticker'] = t                                  
        sub = sub.reset_index().rename(columns={'Date': 'date', 'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'})  
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

# Adds lagged returns to df with different windows
def add_lagged_returns(df: pd.DataFrame, lags: list[int] = [1, 5, 10, 20], price_col: str = 'close') -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=['ticker', 'date'])

    # Same thing as log returns but shift by lag
    for lag in lags:
        df[f'return{lag}d'] = df.groupby('ticker')[price_col].transform(
            lambda x: np.log(x / x.shift(lag))
        )

    return df

# Adds rolling volatility to df with different windows
def add_rolling_volatility(df: pd.DataFrame, windows: list[int] = [5, 10, 20], return_col: str = 'log_return') -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=['ticker', 'date'])

    for w in windows:
        col_name = f'vol_{w}d'

        df[col_name] = df.groupby('ticker')[return_col].transform(
            lambda x: x.rolling(window=w).std() * np.sqrt(252)
        )
    
    return df

# Rolling z score based on close with different windows
def add_rolling_z_score(df: pd.DataFrame, windows: list[int] = [10, 20, 60], return_col: str = 'close') -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=['ticker', 'date'])

    for w in windows:
        col_name = f'z_{w}d'

        df[col_name] = df.groupby('ticker')[return_col].transform(
            lambda x: (x-x.rolling(window=w).mean())/(x.rolling(window=w).std())
        )
    
    return df

# Adds RSI to df with different windows
def add_rsi(df: pd.DataFrame, windows: list[int] = [9, 7, 14, 25], return_col: str = 'close') -> pd.DataFrame:
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    for w in windows:
        col_name = f'rsi_{w}d'
        
        avg_gain = gain.groupby(df['ticker']).transform(lambda x: x.ewm(com=w-1, adjust=False).mean())
        avg_loss = loss.groupby(df['ticker']).transform(lambda x: x.ewm(com=w-1, adjust=False).mean())

        df[col_name] = df.groupby('ticker')[return_col].transform(
            lambda x: 100 - (100 / (1 + (avg_gain / avg_loss)))
        )

    return df
