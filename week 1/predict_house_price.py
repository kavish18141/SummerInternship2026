"""
predict_house_price.py
─────────────────────
A simple command-line tool to predict California median house values
using the saved LinearRegression model (linear_regression_model.pkl).

Usage:
    python predict_house_price.py
"""

import joblib
import numpy as np
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "linear_regression_model.pkl")

FEATURES = {
    "MedInc":     ("Median household income in the block (e.g. 5.0 = $50,000)", 0.5, 15.0),
    "HouseAge":   ("Median house age in years (e.g. 20)", 1, 52),
    "AveRooms":   ("Average number of rooms per household (e.g. 5.5)", 1, 10),
    "AveBedrms":  ("Average number of bedrooms per household (e.g. 1.1)", 1, 5),
    "Population": ("Block group population (e.g. 1200)", 3, 35682),
    "AveOccup":   ("Average household occupancy (e.g. 3.0)", 0.5, 10),
    "Latitude":   ("Block latitude (California: 32.5 – 42.0)", 32.5, 42.0),
    "Longitude":  ("Block longitude (California: -124.4 – -114.3)", -124.4, -114.3),
}


def get_float(prompt: str, lo: float, hi: float) -> float:
    while True:
        try:
            val = float(input(f"  {prompt}: "))
            if lo <= val <= hi:
                return val
            print(f"    ⚠  Please enter a value between {lo} and {hi}.")
        except ValueError:
            print("    ⚠  Invalid input — please enter a number.")


def main():
    print("\n" + "=" * 55)
    print("  🏡  California House Price Predictor")
    print("  Linear Regression Model · Task 1")
    print("=" * 55)

    if not os.path.exists(MODEL_PATH):
        print(f"\n❌  Model file not found at: {MODEL_PATH}")
        print("   Make sure 'linear_regression_model.pkl' is in the same folder.")
        return

    model = joblib.load(MODEL_PATH)
    print("\nModel loaded successfully.\n")

    while True:
        print("Enter block group details (press Ctrl+C to quit):\n")
        values = {}
        for feat, (desc, lo, hi) in FEATURES.items():
            values[feat] = get_float(f"{feat} — {desc}", lo, hi)

        X = pd.DataFrame([values])
        prediction = model.predict(X)[0]
        dollar_value = prediction * 100_000

        print("\n" + "─" * 55)
        print(f"  Predicted Median House Value: ${dollar_value:,.0f}")
        print("─" * 55 + "\n")

        again = input("Predict another? (y/n): ").strip().lower()
        if again != "y":
            print("\nGoodbye! 👋\n")
            break


if __name__ == "__main__":
    main()
