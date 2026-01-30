import yfinance as yf
import pandas as pd

def get_data():
    """Downloads and cleans stock/macro data."""
    print("⬇️ [1/4] Downloading Data...")
    tickers = {'SPY': 'S&P500', '^VIX': 'VIX_Volatility', 
               '^TNX': '10Y_Treasury', 'CL=F': 'Oil_Price', 'GC=F': 'Gold_Price'}
    
    df = yf.download(list(tickers.keys()), start="2010-01-01", end="2024-01-01", auto_adjust=True)
    
    # Handle MultiIndex
    try:
        df = df['Close']
    except KeyError:
        pass # Already flat
        
    df.rename(columns=tickers, inplace=True)
    
    # Imputation: Forward Fill (Justified for Time-Series)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # Feature Engineering: Target = Next Quarter Return
    df['Target_Return'] = df['S&P500'].shift(-63) / df['S&P500'] - 1
    
    return df.dropna()

if __name__ == "__main__":
    # Test this script independently
    df = get_data()
    print(df.head())
    df.to_csv("processed_data.csv")