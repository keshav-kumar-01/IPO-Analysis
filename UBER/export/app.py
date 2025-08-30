# app.py 🚖 Uber Full Analytics Dashboard

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import os

# ------------------------------
# 1. Page Config
# ------------------------------
st.set_page_config(
    page_title="Uber Analytics Dashboard",
    page_icon="🚖",
    layout="wide"
)

st.title("🚖 Uber Analytics Dashboard")

# ------------------------------
# 2. Load Data Helpers
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # current folder

def load_json(filename):
    path = os.path.join(BASE_DIR, filename)
    with open(path, "r") as f:
        return json.load(f)

def load_csv(filename):
    path = os.path.join(BASE_DIR, filename)
    return pd.read_csv(path)

# ------------------------------
# 3. Load All Data
# ------------------------------
try:
    # JSON files
    navigation = load_json("navigation.json")
    ops_index = load_json("ops_index.json")
    fun_stories = load_json("fun_stories.json")
    driver_workload = load_json("driver_workload.json")
    zone_hour_metrics = load_json("zone_hour_metrics.json")
    customers_index = pd.DataFrame(load_json("customers_index.json"))
    daily_metrics = pd.DataFrame(load_json("daily_metrics.json"))

    # CSV files
    payment_stats_by_zone = load_csv("payment_stats_by_zone.csv")
    customer_clv = load_csv("customer_clv.csv")        # adjust if xlsx
    carbon_by_user = load_csv("carbon_by_user.csv")    # adjust if xlsx

    # Map payment codes → names
    payment_method_map = {
        0: "Cash",
        1: "Credit Card",
        2: "Debit Card",
        3: "UPI",
        4: "Wallet",
        5: "Other"
    }
    if "Payment Method" in payment_stats_by_zone.columns:
        payment_stats_by_zone["Payment Method"] = payment_stats_by_zone["Payment Method"].map(payment_method_map)

except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# ------------------------------
# 4. Sidebar Navigation
# ------------------------------
menu = [s["name"] for s in navigation["sections"]]
menu.append("Why This Dashboard Matters")  # New page
choice = st.sidebar.radio("📌 Navigation", menu)

# ------------------------------
# 5. Dashboard Sections
# ------------------------------

# 🔹 Overview
if choice == "User Reports":
    st.header("📊 Operations KPIs")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rides", f"{ops_index['total_rides']:,}")
    col2.metric("Revenue", f"₹{ops_index['total_revenue']:,.0f}")
    col3.metric("Cancellation Rate", f"{ops_index['overall_cancellation_rate']*100:.1f}%")
    col4.metric("Avg Rating", ops_index["avg_rating"])

    st.subheader("📈 Daily Metrics Trend")
    fig = px.line(daily_metrics, x="Date", y="rides", title="Rides per Day")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(daily_metrics, x="Date", y="revenue", title="Revenue per Day")
    st.plotly_chart(fig2, use_container_width=True)

# 🔹 Customer CLV
elif choice == "Customer CLV":
    st.header("👤 Customer Lifetime Value (CLV)")
    st.dataframe(customer_clv)

    # Match actual column name
    if "monetary" in customer_clv.columns:
        fig = px.histogram(customer_clv, x="monetary", nbins=40, title="CLV Distribution")
        st.plotly_chart(fig, use_container_width=True)

# 🔹 Driver Workload
elif choice == "Driver Workload":
    st.header("🚕 Driver Workload (Zone-Hour Analysis)")
    summary = driver_workload["summary"]
    st.write(f"**Total Zone-Hours:** {summary['total_zone_hours']}")
    st.write(f"**Avg Rides per Driver Proxy:** {summary['avg_rides_per_driver_proxy']}")
    st.write(f"**Overall Cancellation Rate:** {summary['overall_cancellation_rate']}")

    zone_df = pd.DataFrame(driver_workload["zone_hour_records"])
    st.dataframe(zone_df)

    fig = px.density_heatmap(
        zone_df,
        x="hour",
        y="Pickup Location",
        z="rides",
        title="Heatmap of Rides by Hour & Location",
        nbinsx=24,
        nbinsy=len(zone_df["Pickup Location"].unique())
    )
    st.plotly_chart(fig, use_container_width=True)

# 🔹 Payment Explorer
elif choice == "Payment Explorer":
    st.header("💳 Payment Stats by Zone")
    st.dataframe(payment_stats_by_zone)

    # Total revenue breakdown
    fig = px.bar(
        payment_stats_by_zone,
        x="Pickup Location",
        y="total_revenue",
        color="Payment Method",
        barmode="stack",
        title="Total Revenue by Zone & Payment Method"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Average fare by Payment Method
    fig2 = px.box(
        payment_stats_by_zone,
        x="Payment Method",
        y="avg_fare",
        points="all",
        title="Average Fare Distribution by Payment Method"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Revenue share by Payment Method
    revenue_share = (
        payment_stats_by_zone.groupby("Payment Method")["total_revenue"].sum().reset_index()
    )
    fig3 = px.pie(
        revenue_share,
        values="total_revenue",
        names="Payment Method",
        title="Revenue Share by Payment Method"
    )
    st.plotly_chart(fig3, use_container_width=True)

# 🔹 Carbon Impact
elif choice == "Carbon Impact":
    st.header("🌱 Carbon Footprint by User")

    # Pagination for huge datasets
    rows_per_page = 100
    total_rows = len(carbon_by_user)
    total_pages = (total_rows // rows_per_page) + 1

    page = st.number_input("Page", min_value=1, max_value=total_pages, step=1)
    start = (page - 1) * rows_per_page
    end = start + rows_per_page

    st.dataframe(carbon_by_user.iloc[start:end])

    # Sample data for plotting (avoid hang)
    sample_df = carbon_by_user.sample(min(1000, len(carbon_by_user)))
    fig = px.histogram(sample_df, x="CO2_emission_kg", nbins=40, title="Distribution of User Carbon Emissions")
    st.plotly_chart(fig, use_container_width=True)

# 🔹 Fun Stories
elif choice == "Recommendations":
    st.header("🎉 Fun Customer Stories")
    for story in fun_stories:
        st.info(story["story"])

# 🔹 Surge Explorer
elif choice == "Surge Explorer":
    st.header("⚡ Surge Explorer (from zone_hour_metrics)")
    zone_df = pd.DataFrame(zone_hour_metrics)
    st.dataframe(zone_df)

    if "total_bookings" in zone_df.columns:
        fig = px.line(
            zone_df,
            x="hour",
            y="total_bookings",
            color="pickup_zone",
            title="Surge Bookings by Hour & Zone"
        )
        st.plotly_chart(fig, use_container_width=True)

# 🔹 Why This Dashboard Matters
elif choice == "Why This Dashboard Matters":
    st.header("💡 Why This Dashboard Matters")

    st.markdown("""
    This Uber Analytics Dashboard is not just numbers and graphs — it’s a **decision-making tool**.  

    ### ✅ For the Company:
    - Optimize pricing and driver allocation during peak hours (via Surge Explorer).
    - Reduce cancellations by analyzing **Driver Workload** and improving incentive plans.
    - Identify high-value customers using **Customer CLV**.

    ### ✅ For Drivers:
    - Know when and where demand is highest.
    - Reduce idle hours and maximize income.
    - Use zone/hour insights to improve route planning.

    ### ✅ For Customers:
    - Benefit from reduced wait times (through better driver allocation).
    - More transparent payment and pricing insights.
    - Support eco-friendly initiatives by tracking carbon impact.

    ### ✅ For Policymakers & Society:
    - Monitor traffic congestion trends.
    - Promote sustainable travel using **Carbon Footprint** data.
    - Ensure fair ride distribution across zones.

    ---
    🌍 In short: **Better Rides, Smarter Business, Greener Planet.**
    """)

# ------------------------------
# Footer
# ------------------------------
st.sidebar.markdown("---")
st.sidebar.success("✅ Dashboard Loaded Successfully")
