# Robo-Advisor: Macroeconomic S&P 500 Predictor

A modular Machine Learning pipeline designed to predict the **Next Quarter's S&P 500 Returns** using macroeconomic leading indicators. This project demonstrates data ingestion, exploratory statistical analysis, and model evaluation (Linear Regression vs. Random Forest vs. Gradient Boosting).

---

## Project Overview
The **Robo-Advisor** aims to remove emotional bias from investing by providing a data-driven "Buy/Hold/Sell" signal based on the current economic regime. 

### Key Features
* **Target:** Cumulative returns for the next 63 trading days (approx. one quarter).
* **Predictors:**
    * **VIX Index:** Market sentiment and fear levels.
    * **10-Year Treasury Yield:** Interest rate pressure (the "gravity" of finance).
    * **Crude Oil Prices:** Proxy for global demand and inflationary pressure.
    * **Gold Prices:** Safe-haven demand and geopolitical risk proxy.

---

## Project Structure
The project is built modularly to ensure scalability and follow software engineering best practices.

* data_loader.py: Handles raw data ingestion from Yahoo Finance and feature engineering.
* eda.py: Performs statistical tests (ADF Test) and generates visualization charts.
* model.py: Compares three ML models and evaluates them using **RMSE** and **R²**.
* main.py: The master controller that runs the end-to-end pipeline.

---

## Getting Started

1. Installation
Ensure you have an Anaconda environment activated, then install the dependencies:

**bash**
pip install yfinance pandas numpy seaborn matplotlib statsmodels scikit-learn

2. Running the Pipeline
To run the full analysis and generate a recommendation, execute the master script

**bash**
python main.py

2. Data Cleaning
We utilized Forward Filling for missing financial data. We avoided mean imputation to prevent Data Leakage, ensuring that the model never "looks into the future" during the training phase.

3. Statistical Rigor
We conducted an Augmented Dickey-Fuller (ADF) Test to ensure the stationarity of our target variable. A stationary target is essential for reliable regression results.

4. Model Performance

     Model            RMSE
Linear Regression -   0.0900
Random Forest     -   0.0875
Gradient Boosting -   0.1136

5. Insight: 
Random Forest outperformed Linear Regression because the relationship between macro features and market returns is non-linear and subject to threshold effects. 

6. Business Conclusion
This model triggers "SELL/HEDGE" recommendation. The Robo-Advisor demonstrates that while the stock market contains significant noise, macroeconomic indicators explain a meaningful portion of mid-term variance. By monitoring the VIX and Bond Yields, the model provides a statistical edge in portfolio risk management.

