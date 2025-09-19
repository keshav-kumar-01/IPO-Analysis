# ============================
# Step 8: Investor Strategy Simulation (Using Saved Models)
# ============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ----------------------------
# User Settings
# ----------------------------
DATA_PATH = r"D:\IPO_analysis\data\ipo_features.csv"
MODEL_PATH = r"D:\IPO_analysis\models\best_xgb_model.pkl"

# ----------------------------
# Load Data
# ----------------------------
df = pd.read_csv(DATA_PATH)

# ----------------------------
# Load Model (handle dict or model directly)
# ----------------------------
with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

if isinstance(saved, dict):
    model = saved["model"]
    trained_features = saved.get("features", None)
else:
    model = saved
    trained_features = model.get_booster().feature_names

# ----------------------------
# Select features
# ----------------------------
if trained_features is None:
    selected_features = [
        'Issue_Size(crores)', 'QIB', 'HNI', 'RII', 'Total',
        'Offer Price', 'List Price', 'Listing Gain', 'CMP(BSE)', 'CMP(NSE)',
        'Current Gains', 'Year', 'Month', 'Quarter', 'High_QIB_Demand',
        'High_HNI_Demand', 'High_RII_Demand', 'High_Total_Demand',
        'Log_Issue_Size', 'Price_to_List_Ratio', 'Price_to_CMP_BSE',
        'Price_to_CMP_NSE', 'IPO_in_Q1', 'IPO_in_Q2', 'IPO_in_Q3', 'IPO_in_Q4'
    ]
else:
    selected_features = trained_features

X = df[selected_features]

# ----------------------------
# Preprocessing
# ----------------------------
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)

# ----------------------------
# Model Predictions
# ----------------------------
df["Predicted_Wealth_Creator"] = model.predict(X_scaled)
df["Prediction_Prob"] = model.predict_proba(X_scaled)[:, 1]

# ----------------------------
# Strategy Functions
# ----------------------------
capital = 10_00_000  # 10 lakhs
invest_per_ipo = 50_000

def strategy_offer_hold(row):
    """Buy at Offer Price and hold till CMP(BSE)."""
    if row["Offer Price"] > 0 and row["CMP(BSE)"] > 0:
        return invest_per_ipo * (row["CMP(BSE)"] / row["Offer Price"])
    return invest_per_ipo

def strategy_listing_gain(row):
    """Buy at Offer Price and sell on Day-1 (Listing Price)."""
    if row["Offer Price"] > 0 and row["List Price"] > 0:
        return invest_per_ipo * (row["List Price"] / row["Offer Price"])
    return invest_per_ipo

def strategy_model_based(row):
    """Buy only if model predicts wealth creator, exit on listing gain."""
    if row["Predicted_Wealth_Creator"] == 1 and row["Offer Price"] > 0 and row["List Price"] > 0:
        return invest_per_ipo * (row["List Price"] / row["Offer Price"])
    return invest_per_ipo

def strategy_cap_weighted(row, total_issue_size):
    """Allocate proportionally to issue size, sell on CMP."""
    if row["Offer Price"] > 0 and row["CMP(BSE)"] > 0:
        alloc = capital * (row["Issue_Size(crores)"] / total_issue_size)
        return alloc * (row["CMP(BSE)"] / row["Offer Price"])
    return 0

# ----------------------------
# Apply Strategies
# ----------------------------
df["Offer_Hold"] = df.apply(strategy_offer_hold, axis=1)
df["Listing_Gain"] = df.apply(strategy_listing_gain, axis=1)
df["Model_Based"] = df.apply(strategy_model_based, axis=1)

total_issue_size = df["Issue_Size(crores)"].sum()
df["Cap_Weighted"] = df.apply(lambda r: strategy_cap_weighted(r, total_issue_size), axis=1)

# ----------------------------
# Evaluate Results
# ----------------------------
strategies = ["Offer_Hold", "Listing_Gain", "Model_Based", "Cap_Weighted"]

results = {}
for strat in strategies:
    final_val = df[strat].sum()
    roi = (final_val - capital) / capital * 100
    win_rate = (df[strat] > invest_per_ipo).mean() * 100
    results[strat] = {"Final Value": final_val, "ROI %": roi, "Win Rate %": win_rate}

print("\n===== Strategy Results =====")
for strat, vals in results.items():
    print(f"{strat}: Final Value ₹{vals['Final Value']:.2f}, ROI {vals['ROI %']:.2f}%, Win Rate {vals['Win Rate %']:.2f}%")

# ----------------------------
# Visualization
# ----------------------------
plt.figure(figsize=(12, 7))
for strat in strategies:
    plt.plot(np.cumsum(df[strat]), label=strat)

plt.xlabel("IPO Number")
plt.ylabel("Portfolio Value (₹)")
plt.title("Investor Strategy Simulations")
plt.legend()
plt.grid(True)
plt.show()
