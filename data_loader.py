import yfinance as yf
import pandas as pd
import os

def get_data():
    """Downloads and cleans stock/macro data."""
    print("[1/4] Downloading Data...")
    
    tickers = {
        'SPY': 'S&P500', 
        '^VIX': 'VIX_Volatility', 
        '^TNX': '10Y_Treasury', 
        'CL=F': 'Oil_Price', 
        'GC=F': 'Gold_Price'
    }
    
    # Download data
    df = yf.download(list(tickers.keys()), start="2010-01-01", end="2024-01-01", auto_adjust=True)
    
    # Handle the "MultiIndex" issue common with new yfinance versions
    try:
        # If 'Close' is a top-level column, grab it
        if 'Close' in df.columns.levels[0]:
            df = df['Close']
    except AttributeError:
        # If it's already flat (older versions or single ticker), just copy
        pass
        
    # Rename columns
    # We strip the ticker symbols to match our friendly names
    df.rename(columns=tickers, inplace=True)
    
    # Imputation: Forward Fill
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # Feature Engineering
    df['Target_NextQ_Return'] = df['S&P500'].shift(-63) / df['S&P500'] - 1
    
    df_clean = df.dropna()
    print(f"   Data Processed. Shape: {df_clean.shape}")
    return df_clean

if __name__ == "__main__":
    # 1. Run the function
    df = get_data()
    
    # 2. Define the filename explicitly
    filename = "cleaned_stock_data.csv"
    
    # 3. Save it
    df.to_csv(filename)
    
    # 4. PROOF: Print exactly where it is
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, filename)
    
    if os.path.exists(file_path):
        print(f"\nSUCCESS! File saved at:\n{file_path}")
        print("You can now run 'eda.py' or your notebook.")
    else:
        print("Error: File was not saved. Check permissions.")