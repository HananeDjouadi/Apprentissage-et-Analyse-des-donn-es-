import pickle
import numpy as np

# 1. Load the pipeline
filename = "credit_scoring_pipeline.pkl"
print(f"Loading {filename}...")

with open(filename, 'rb') as f:
    model = pickle.load(f)

print("Model loaded successfully!")

# 2. Create dummy data for testing
# We must provide ALL original variables (13 columns), 
# even if the pipeline selects only 8 later.
# Example values: Seniority, Home, Time, Age, Marital, Records, Job, Expenses, Income, Assets, Debt, Amount, Price
input_data = np.array([[
    5,   # Seniority
    1,   # Home
    36,  # Time
    45,  # Age
    2,   # Marital
    1,   # Records
    3,   # Job
    50,  # Expenses
    150, # Income
    3000,# Assets
    0,   # Debt
    1000,# Amount
    1200 # Price
]])

# 3. Predict
prediction = model.predict(input_data)
proba = model.predict_proba(input_data)

print(f"\nPrediction: {prediction[0]}")
print(f"Probability (Class 0 vs Class 1): {proba[0]}")