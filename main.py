# Import functions from other files
from data_loader import get_data
from eda import run_eda
from model import train_model, predict_next_quarter

def main():
    print("--- ROBO ADVISOR PIPELINE STARTED ---")
    
    # Step 1: Get Data
    df = get_data()
    
    # Step 2: Analyze
    run_eda(df)
    
    # Step 3: Train
    model, latest_row = train_model(df)
    
    # Step 4: Decide
    decision = predict_next_quarter(model, latest_row)
    print(f"Final Recommendation: {decision}")
    
    print("--- PIPELINE FINISHED ---")

if __name__ == "__main__":
    main()