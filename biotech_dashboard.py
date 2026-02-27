import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Executive Biotech Manufacturing Dashboard", 
layout="wide")

st.title("🧬 Executive Biotech Manufacturing Dashboard")

# Database Connection
engine = create_engine(
    
"postgresql://postgres:newpassword123@localhost:5432/biotech_manufacturing"
)

# Load Data
query = """
SELECT batch_id, status, actual_yield
FROM batches;
"""

df = pd.read_sql(query, engine)

# Sidebar Filter
st.sidebar.header("Filter Options")
status_filter = st.sidebar.selectbox(
    "Select Batch Status",
    options=["All"] + list(df["status"].unique())
)

if status_filter != "All":
    df = df[df["status"] == status_filter]

# KPI Calculations
total_batches = len(df)
avg_yield = round(df["actual_yield"].mean(), 2)
success_rate = round(
    (len(df[df["status"] == "Completed"]) / total_batches) * 100
    if total_batches > 0 else 0,
    2
)

col1, col2, col3 = st.columns(3)
col1.metric("Total Batches", total_batches)
col2.metric("Average Yield (%)", avg_yield)
col3.metric("Success Rate (%)", success_rate)

st.markdown("---")

# Yield Distribution
st.subheader("Yield Distribution")
fig, ax = plt.subplots()
ax.hist(df["actual_yield"], bins=10)
ax.set_xlabel("Actual Yield (%)")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Predictive Model
if len(df) > 1:
    df["batch_index"] = np.arange(len(df))
    X = df[["batch_index"]]
    y = df["actual_yield"]

    model = LinearRegression()
    model.fit(X, y)

    next_batch_index = np.array([[len(df)]])
    predicted_yield = model.predict(next_batch_index)[0]

    st.markdown("### 🔮 Predicted Next Batch Yield")
    st.metric("Predicted Yield (%)", round(predicted_yield, 2))

# Anomaly Detection
st.markdown("### 🚨 Yield Anomaly Detection")
mean_yield = df["actual_yield"].mean()
std_yield = df["actual_yield"].std()

df["anomaly"] = np.where(
    (df["actual_yield"] > mean_yield + 2*std_yield) |
    (df["actual_yield"] < mean_yield - 2*std_yield),
    "Anomaly",
    "Normal"
)

anomalies = df[df["anomaly"] == "Anomaly"]
st.write(f"Detected {len(anomalies)} anomalous batches")

if len(anomalies) > 0:
    st.dataframe(anomalies)

st.subheader("Detailed Batch Data")
st.dataframe(df)
