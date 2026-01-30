import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def run_eda(df):
    """Generates visualizations and stats."""
    print("\n [2/4] Running Exploratory Analysis...")
    
    # Chart 1: Heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("eda_heatmap.png") # Save instead of showing
    print("   > Saved 'eda_heatmap.png'")
    
    # Stat Test: ADF
    result = adfuller(df['Target_Return'])
    print(f"   > ADF P-Value: {result[1]:.5f}")
    if result[1] < 0.05:
        print("   > Data is Stationary (Ready for ML)")
    else:
        print("   > Data is Non-Stationary")