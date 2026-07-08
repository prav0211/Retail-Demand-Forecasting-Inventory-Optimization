import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Retail Demand Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Retail Demand Forecasting & Inventory Optimization")

# Load Dataset
df = pd.read_csv("retail_demand_dataset.csv")

# Sidebar Filters
st.sidebar.header("Filters")

store = st.sidebar.selectbox(
    "Select Store",
    ["All"] + list(df["Store"].unique())
)

department = st.sidebar.selectbox(
    "Select Department",
    ["All"] + list(df["Department"].unique())
)

category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + list(df["Category"].unique())
)

# Apply Filters
filtered_df = df.copy()

if store != "All":
    filtered_df = filtered_df[filtered_df["Store"] == store]

if department != "All":
    filtered_df = filtered_df[filtered_df["Department"] == department]

if category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == category]

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sales", int(filtered_df["Sales"].sum()))
col2.metric("Total Inventory", int(filtered_df["Inventory"].sum()))
col3.metric("Average Price", round(filtered_df["Price"].mean(), 2))
col4.metric("Average Forecast", int(filtered_df["Demand_Forecast"].mean()))

st.divider()

# Dataset Preview
st.subheader("Retail Dataset")
st.dataframe(filtered_df)

# Sales Trend
st.subheader("Sales Trend")

sales = filtered_df.groupby("Date")["Sales"].sum()

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(sales.index, sales.values)
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.set_title("Daily Sales")
st.pyplot(fig)

# Category-wise Sales
st.subheader("Category-wise Sales")

category_sales = filtered_df.groupby("Category")["Sales"].sum()

fig, ax = plt.subplots(figsize=(8,4))
ax.bar(category_sales.index, category_sales.values)
plt.xticks(rotation=45)
st.pyplot(fig)

# Inventory
st.subheader("Inventory by Department")

inventory = filtered_df.groupby("Department")["Inventory"].sum()

fig, ax = plt.subplots(figsize=(8,4))
ax.bar(inventory.index, inventory.values)
plt.xticks(rotation=45)
st.pyplot(fig)

# Download Dataset
st.download_button(
    "Download Report",
    filtered_df.to_csv(index=False),
    file_name="Retail_Report.csv",
    mime="text/csv"
)
