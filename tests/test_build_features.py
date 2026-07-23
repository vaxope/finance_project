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

    df = add_log_returns(df)
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

def test_rolling_volatility_calc():
    df = pd.DataFrame({
        'ticker': ['AAA', 'AAA', 'AAA', 'AAA', 'AAA'],
        'date': pd.date_range('2023-01-01', periods=5),
        'close': [100.0, 110.0, 121.0, 133.1, 150.0] 
    })
    
    df = add_log_returns(df)
    result = add_rolling_volatility(df, windows=[1, 2])

    # vol_1 should all be na because std has a denominator of n - 1, and 1 - 1 = 0
    assert pd.isna(result["vol_1d"].iloc[0])
    assert pd.isna(result["vol_1d"].iloc[1])
    assert pd.isna(result["vol_1d"].iloc[2])
    assert pd.isna(result["vol_1d"].iloc[3])
    assert pd.isna(result["vol_1d"].iloc[4])

    # Row 0 has no return and row 1 has 1 return, so both should be empty
    assert pd.isna(result["vol_2d"].iloc[0])
    assert pd.isna(result["vol_2d"].iloc[1])

    assert np.isclose(result["vol_2d"].iloc[2], 0.0, atol=1e-9)
    assert np.isclose(result["vol_2d"].iloc[3], 0.0, atol=1e-9)
    
    expected_vol = (np.abs(np.log(133.1/121.0)-np.log(150.0/133.1))/np.sqrt(2)) * np.sqrt(252)
    assert np.isclose(result["vol_2d"].iloc[4], expected_vol, atol=1e-6) 

def test_rolling_volatility_cross_ticker_leakage():
    df = pd.DataFrame({
        'ticker': ['AAA', 'AAA', 'AAA', 'BBB', 'BBB', 'BBB'],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03']),
        'close': [100.0, 110.0, 121.0, 50.0, 55.0, 60.5] 
    })
    
    df = add_log_returns(df)
    result = add_rolling_volatility(df, windows=[2])
    
    bbb_rows = result[result['ticker'] == 'BBB']
    assert pd.isna(bbb_rows["vol_2d"].iloc[0])
    assert pd.isna(bbb_rows["vol_2d"].iloc[1])
    assert np.isclose(bbb_rows["vol_2d"].iloc[2], 0.0, atol=1e-9)

def test_rolling_z_score_calc():
    df = pd.DataFrame({
            'ticker': ['AAA', 'AAA', 'AAA', 'AAA', 'AAA'],
            'date': pd.date_range('2023-01-01', periods=5),
            'close': [100.0, 110.0, 121.0, 133.1, 150.0] 
        })

    df = add_log_returns(df)
    result = add_rolling_z_score(df, windows=[1, 2])

    # z_1d should all be na because std has a denominator of n - 1, and X - mean = 0 and 0/0=nan
    assert pd.isna(result["z_1d"].iloc[0])
    assert pd.isna(result["z_1d"].iloc[1])
    assert pd.isna(result["z_1d"].iloc[2])
    assert pd.isna(result["z_1d"].iloc[3])
    assert pd.isna(result["z_1d"].iloc[4])

    # 2 day lookback window means first is nan
    assert pd.isna(result["z_2d"].iloc[0])
    
    assert np.isclose(result["z_2d"].iloc[1], 1 / np.sqrt(2), atol=1e-6)
    assert np.isclose(result["z_2d"].iloc[2], 1 / np.sqrt(2), atol=1e-6)
    assert np.isclose(result["z_2d"].iloc[3], 1 / np.sqrt(2), atol=1e-6) 
    
    assert np.isclose(result["z_2d"].iloc[4], 1 / np.sqrt(2), atol=1e-6)

def test_rolling_z_score_cross_ticker_leakage():
    df = pd.DataFrame({
        'ticker': ['AAA', 'AAA', 'AAA', 'BBB', 'BBB', 'BBB'],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03']),
        'close': [100.0, 110.0, 121.0, 50.0, 55.0, 65.0]
    })
    df = add_log_returns(df)
    result = add_rolling_z_score(df, windows=[2])
    

    bbb_rows = result[result['ticker'] == 'BBB']
    
    assert pd.isna(bbb_rows["z_2d"].iloc[0])
    assert np.isclose(bbb_rows["z_2d"].iloc[1], 0.70710678, atol=1e-6)    
    assert pd.isna(bbb_rows["z_2d"].iloc[2]) == False 

def test_rsi_calc():
    df = pd.DataFrame({
        'ticker': ['AAA', 'AAA', 'AAA', 'AAA', 'AAA'],
        'date': pd.date_range('2023-01-01', periods=5),
        'close': [100.0, 110.0, 121.0, 133.1, 150.0]
    })
    
    result = add_rsi(df, windows=[1, 2])
    

    assert pd.isna(result["rsi_1d"].iloc[0])

    assert np.isclose(result["rsi_1d"].iloc[1], 100.0, atol=1e-6)
    assert np.isclose(result["rsi_1d"].iloc[2], 100.0, atol=1e-6)
    assert np.isclose(result["rsi_1d"].iloc[3], 100.0, atol=1e-6)
    assert np.isclose(result["rsi_1d"].iloc[4], 100.0, atol=1e-6)
    
    assert pd.isna(result["rsi_2d"].iloc[0])

    assert np.isclose(result["rsi_2d"].iloc[1], 100.0, atol=1e-6)
    assert np.isclose(result["rsi_2d"].iloc[2], 100.0, atol=1e-6)
    assert np.isclose(result["rsi_2d"].iloc[3], 100.0, atol=1e-6)
    assert np.isclose(result["rsi_2d"].iloc[4], 100.0, atol=1e-6)

def test_rsi_cross_ticker_leakage():
    df = pd.DataFrame({
        'ticker': ['AAA', 'AAA', 'BBB', 'BBB'],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02']),
        'close': [100.0, 110.0, 200.0, 50.0] # AAA goes up, BBB goes down
    })
    
    result = add_rsi(df, windows=[1])
    
    bbb_rows = result[result['ticker'] == 'BBB']
    
    assert pd.isna(bbb_rows["rsi_1d"].iloc[0])
    assert np.isclose(bbb_rows["rsi_1d"].iloc[1], 0.0, atol=1e-6)


