import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set non-interactive backend to prevent "No display name" errors
import matplotlib
matplotlib.use('Agg') 

def run_eda(df=None):
    print("Starting EDA Process...")

    # Load data if not passed in
    if df is None:
        try:
            # Try loading the cleaned data
            df = pd.read_csv("cleaned_stock_data.csv")
            # Handle Date index
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif df.index.name != 'Date':
                df.index = pd.to_datetime(df.index)
        except FileNotFoundError:
            print("Error: 'cleaned_stock_data.csv' not found. Run data_loader.py first.")
            return

    # Create the output directory if it doesn't exist
    if not os.path.exists('eda_images'):
        os.makedirs('eda_images')
        print("   > Created folder 'eda_images/'")

    # GRAPH 1: Correlation Heatmap
    plt.figure(figsize=(10, 6))
    # Select numeric columns only to avoid errors
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    # SAVE instead of show
    plt.savefig('eda_images/heatmap.png')
    plt.close() # Close memory
    print(" Saved: eda_images/heatmap.png")

    # GRAPH 2: Market Regime (Price vs VIX)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('S&P 500', color=color)
    ax1.plot(df.index, df['S&P500'], color=color, label='S&P 500')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('VIX (Fear)', color=color)
    ax2.fill_between(df.index, df['VIX_Volatility'], color=color, alpha=0.3, label='VIX')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Market Regime: Price vs Fear')
    plt.tight_layout()
    plt.savefig('eda_images/market_regime.png')
    plt.close()
    print(" Saved: eda_images/market_regime.png")

    # GRAPH 3: Hypothesis Check
    plt.figure(figsize=(10, 6))
    sns.regplot(x='10Y_Treasury', y='Target_NextQ_Return', data=df, 
                scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title('Hypothesis: Rates vs Returns')
    plt.savefig('eda_images/hypothesis_scatter.png')
    plt.close()
    print(" Saved: eda_images/hypothesis_scatter.png")

    # GRAPH 4: Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Target_NextQ_Return'], bins=50, kde=True, color='green')
    plt.title('Return Distribution')
    plt.savefig('eda_images/distribution.png')
    plt.close()
    print(" Saved: eda_images/distribution.png")
    
    print("\n🎉 EDA Complete. Check the 'eda_images' folder!")

if __name__ == "__main__":
    run_eda()