# ============================
# streamlit_app.py
# IPO Analysis Dashboard
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ----------------------------
# App Config (must be first Streamlit command)
# ----------------------------
st.set_page_config(
    page_title="IPO Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# ----------------------------
# Load Data & Model
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        r"D:\IPO_analysis\data\ipo_features.csv",
        parse_dates=["Date"]
    )

@st.cache_resource
def load_model():
    with open(r"D:\IPO_analysis\models\best_xgb_model.pkl", "rb") as f:
        return pickle.load(f)   # dict {model, features, imputer, scaler}

ipo_df = load_data()
model = load_model()

# Extract model components
trained_model = model["model"]
imputer = model["imputer"]
scaler = model["scaler"]
feature_names = model["features"]

# ----------------------------
# App Layout
# ----------------------------
st.title("📊 IPO Analysis & Strategy Dashboard")
st.markdown("Welcome to the interactive dashboard for IPO data exploration, model insights, and strategy simulation.")

# Sidebar Navigation
menu = st.sidebar.radio(
    "📌 Navigation",
    ["Data Explorer", "IPO Detail View", "Strategy Simulator", "Model Playground"]
)

# ==================================================
# 1) Data Explorer
# ==================================================
if menu == "Data Explorer":
    st.header("📂 Data Explorer")
    st.markdown("---")

    # Filters
    years = st.multiselect("📅 Select Year(s)", sorted(ipo_df["Year"].unique()))

    st.subheader("🔎 Demand Flags")
    demand_flags = ["High_QIB_Demand", "High_HNI_Demand", "High_RII_Demand", "High_Total_Demand"]
    active_flags = [flag for flag in demand_flags if st.checkbox(flag, False)]

    st.subheader("💰 Issue Size Filter")
    issue_size = st.slider(
        "Issue Size (₹ Crores)",
        int(ipo_df["Issue_Size(crores)"].min()),
        int(ipo_df["Issue_Size(crores)"].max()),
        (
            int(ipo_df["Issue_Size(crores)"].min()),
            int(ipo_df["Issue_Size(crores)"].max())
        )
    )

    # Apply filters
    df_filtered = ipo_df.copy()
    if years:
        df_filtered = df_filtered[df_filtered["Year"].isin(years)]
    for flag in active_flags:
        df_filtered = df_filtered[df_filtered[flag] == 1]
    df_filtered = df_filtered[
        (df_filtered["Issue_Size(crores)"] >= issue_size[0]) &
        (df_filtered["Issue_Size(crores)"] <= issue_size[1])
    ]

    st.dataframe(df_filtered.head(50), use_container_width=True)

    st.subheader("📊 Distribution Charts")
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df_filtered["Year"].value_counts())
    with col2:
        st.bar_chart(df_filtered["Quarter"].value_counts())

# ==================================================
# 2) IPO Detail View
# ==================================================
elif menu == "IPO Detail View":
    st.header("📈 IPO Detail View")
    st.markdown("---")

    ipo_name = st.selectbox("🏢 Select IPO", ipo_df["IPO_Name"].unique())
    df_ipo = ipo_df[ipo_df["IPO_Name"] == ipo_name]

    st.subheader(f"🔖 IPO Details: {ipo_name}")
    st.dataframe(df_ipo, use_container_width=True)

    if "Listing Gain" in df_ipo.columns:
        st.subheader("📉 Performance Time Series")
        st.line_chart(df_ipo.set_index("Date")["Listing Gain"])

    st.subheader("🧾 Model Explanation (SHAP)")
    try:
        # Preprocess features
        X = ipo_df[feature_names]
        X_proc = scaler.transform(imputer.transform(X))

        ipo_row = df_ipo[feature_names]
        ipo_row_proc = scaler.transform(imputer.transform(ipo_row))

        # SHAP explainer
        explainer = shap.Explainer(trained_model, X_proc, feature_names=feature_names)
        shap_values = explainer(ipo_row_proc)

        # IPO-specific explanation
        st.write(f"📌 SHAP Explanation for **{ipo_name}**")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

        # Global summary
        st.subheader("🌍 Global Feature Importance")
        shap_summary = explainer(X_proc)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.bar(shap_summary, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"⚠️ SHAP explanation not available: {e}")

# ==================================================
# 3) Strategy Simulator
# ==================================================
elif menu == "Strategy Simulator":
    st.header("💰 Strategy Simulator")
    st.markdown("---")

    capital = st.number_input("💵 Initial Capital (₹)", min_value=10000, value=100000, step=10000)
    strategy = st.selectbox(
        "📊 Choose Strategy",
        ["Buy & Hold (Offer → CMP)", "Day-1 Listing Gain", "Model-based Selection"]
    )

    st.info(f"Simulating strategy: **{strategy}** with capital ₹{capital:,}")

    # Dummy backtest logic (placeholder)
    returns = np.random.normal(0.02, 0.05, size=50).cumsum()
    st.line_chart(returns)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Final Portfolio Value", f"₹{capital * (1 + returns[-1]):,.0f}")
    with col2:
        st.metric("Total Return %", f"{returns[-1]*100:.2f}%")

# ==================================================
# 4) Model Playground
# ==================================================
elif menu == "Model Playground":
    st.header("🤖 Model Playground")
    st.markdown("---")

    threshold = st.slider("⚖️ Prediction Threshold", 0.0, 1.0, 0.5, 0.05)

    X = ipo_df[feature_names]
    X_proc = scaler.transform(imputer.transform(X))

    y_true = ipo_df["is_wealth_creator"]
    y_pred_proba = trained_model.predict_proba(X_proc)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False)
    st.pyplot(fig)

    st.subheader("📊 Expected Returns Distribution")
    fig, ax = plt.subplots()
    ax.hist(y_pred_proba, bins=20, edgecolor="black")
    ax.set_title("Predicted Probability Distribution")
    ax.set_xlabel("Probability of Wealth Creator")
    ax.set_ylabel("Count")
    st.pyplot(fig)
