# ============================
# Step 5: Feature Engineering
# ============================

import os
import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new engineered features for IPO dataset
    """

    df = df.copy()

    # --- 1. Oversubscription flags ---
    df["High_QIB_Demand"] = (df["QIB"] > 10).astype(int)   # QIB subscribed >10x
    df["High_HNI_Demand"] = (df["HNI"] > 10).astype(int)
    df["High_RII_Demand"] = (df["RII"] > 10).astype(int)
    df["High_Total_Demand"] = (df["Total"] > 20).astype(int)

    # --- 2. Normalized / log features ---
    df["Log_Issue_Size"] = np.log1p(df["Issue_Size(crores)"])
    df["Price_to_List_Ratio"] = df["List Price"] / df["Offer Price"]
    df["Price_to_CMP_BSE"] = df["CMP(BSE)"] / df["Offer Price"]
    df["Price_to_CMP_NSE"] = df["CMP(NSE)"] / df["Offer Price"]

    # --- 3. Temporal features ---
    df["IPO_in_Q1"] = (df["Quarter"] == 1).astype(int)
    df["IPO_in_Q2"] = (df["Quarter"] == 2).astype(int)
    df["IPO_in_Q3"] = (df["Quarter"] == 3).astype(int)
    df["IPO_in_Q4"] = (df["Quarter"] == 4).astype(int)

    # --- 4. Targets ---
    df["is_listing_win"] = (df["Listing Gain"] > 0).astype(int)
    df["is_big_listing_win"] = (df["Listing Gain"] > 10).astype(int)
    df["is_wealth_creator"] = (df["Current Gains"] > 50).astype(int)
    df["is_loss_maker"] = (df["Current Gains"] < 0).astype(int)

    return df


if __name__ == "__main__":
    # --- Safe path handling ---
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root (.. from src/)
    INPUT_PATH = os.path.join(BASE_DIR, "data", "ipo_clean.csv")
    OUTPUT_PATH = os.path.join(BASE_DIR, "data", "ipo_features.csv")

    # Load dataset
    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])

    # Feature engineering
    df = add_features(df)

    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Feature-engineered dataset saved to {OUTPUT_PATH}")
