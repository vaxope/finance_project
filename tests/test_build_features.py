import yfinance as yf
import numpy as np
import pandas as pd

from src.data.features.build_features import (
    load_prices_long,
    add_log_returns,
    add_lagged_returns,
    add_rolling_volatility,
    add_rolling_z_score,
    add_rsi,
)

def test_log_returns_calc():
    df = pd.DataFrame({
        'ticker': ['AAA', 'AAA', 'AAA'],
        'date': pd.date_range('2023-01-01', periods=3),
        'close': [100.0, 110.0, 121.0] # 10% increase each day
    })

    result = add_log_returns(df)

    # First row should be na since it doesn't have anything to compare to
    assert pd.isna(result['log_return'].iloc[0])

    # Compares expected value to actual calculation
    assert np.isclose(result['log_return'].iloc[1], np.log(110.0 / 100.0))
    assert np.isclose(result['log_return'].iloc[2], np.log(121.0 / 110.0))

def test_log_return_cross_ticker_leakage():
    df = pd.DataFrame({
        'ticker': ['AAA', 'AAA', 'BBB', 'BBB'],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02']),
        'close': [100.0, 200, 25.0, 30.0] 
    })

    result = add_log_returns(df)

    # Closing value from AAA should not affect BBB's first day log return 
    bbb_first_row = result[result['ticker'] == 'BBB'].iloc[0]
    assert pd.isna(bbb_first_row['log_return'])

def test_lagged_return_calc():
    df = pd.DataFrame({
        'ticker': ['AAA', 'AAA', 'AAA', 'AAA'],
        'date': pd.date_range('2023-01-01', periods=4),
        'close': [100.0, 110.0, 121.0, 133.1] # 10% increase each day
    })

    result = add_lagged_returns(df, lags=[1, 2])

    # First row for 1d should NA and first two should be NA for 2d lag
    assert pd.isna(result["return1d"].iloc[0])
    assert pd.isna(result["return2d"].iloc[0])
    assert pd.isna(result["return2d"].iloc[1])

    # Compares expected to actual values for 1d
    assert np.isclose(result["return1d"].iloc[1], np.log(110.0 / 100.0))
    assert np.isclose(result["return1d"].iloc[2], np.log(121.0 / 110.0))

    # Compares expected to actual values for 2d
    assert np.isclose(result["return2d"].iloc[2], np.log(121.0 / 100.0))
    assert np.isclose(result["return2d"].iloc[3], np.log(133.1 / 110.0))

def test_lagged_return_cross_ticker_leakage():
    df = pd.DataFrame({
        'ticker': ['AAA', 'AAA', 'BBB', 'BBB'],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02']),
        'close': [100.0, 200, 25.0, 30.0] 
    })

    result = add_lagged_returns(df, lags=[1])
    bbb_first_row = result[result['ticker'] == 'BBB'].iloc[0]
    assert pd.isna(bbb_first_row['return1d'])

    bbb_second_row = result[result['ticker'] == 'BBB'].iloc[1]
    assert np.isclose(bbb_second_row['return1d'], np.log(30.0 / 25.0))