import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_model(df):
    """Trains Random Forest and returns the best model."""
    print("\n🤖 [3/4] Training Model...")
    
    X = df[['VIX_Volatility', '10Y_Treasury', 'Oil_Price', 'Gold_Price']]
    y = df['Target_Return']
    
    # Time-Series Split (No Shuffling)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Grid Search
    param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=tscv)
    grid.fit(X_train, y_train)
    
    # Evaluate
    preds = grid.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"   > Best Params: {grid.best_params_}")
    print(f"   > RMSE: {rmse:.4f}")
    
    return grid.best_estimator_, X_test.iloc[[-1]] # Return model + latest data

def predict_next_quarter(model, latest_data):
    """Makes the final Buy/Sell call."""
    pred = model.predict(latest_data)[0]
    print(f"\n [4/4] Prediction for Next Quarter: {pred:.2%}")
    
    if pred > 0.05: return "STRONG BUY"
    elif pred > 0.015: return "HOLD / ACCUMULATE"
    else: return "SELL / HEDGE"