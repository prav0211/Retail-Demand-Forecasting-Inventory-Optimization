# ==========================================================
# Retail Demand Forecasting Project
# Week 2-3: Data Cleaning and Exploratory Data Analysis (EDA)
# ==========================================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Improve plot appearance
plt.style.use("ggplot")

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
df = pd.read_csv("retail_demand_dataset.csv")

print("========== First 5 Rows ==========")
print(df.head())

print("\n========== Dataset Information ==========")
print(df.info())

print("\n========== Statistical Summary ==========")
print(df.describe())

# ----------------------------------------------------------
# Check Missing Values
# ----------------------------------------------------------
print("\n========== Missing Values ==========")
print(df.isnull().sum())

# Fill Missing Values
numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# ----------------------------------------------------------
# Check Duplicate Rows
# ----------------------------------------------------------
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove Duplicate Rows
df = df.drop_duplicates()

# ----------------------------------------------------------
# Convert Date Column
# ----------------------------------------------------------
df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# ==========================================================
# Exploratory Data Analysis
# ==========================================================

# 1. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()

# ----------------------------------------------------------

# 2. Demand Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Demand"],
             bins=25,
             kde=True)

plt.title("Demand Distribution")
plt.xlabel("Demand")
plt.show()

# ----------------------------------------------------------

# 3. Demand Outliers
plt.figure(figsize=(6,5))
sns.boxplot(y=df["Demand"])

plt.title("Demand Boxplot")
plt.show()

# ----------------------------------------------------------

# 4. Price vs Demand
plt.figure(figsize=(8,5))
sns.scatterplot(data=df,
                x="Price",
                y="Demand")

plt.title("Price vs Demand")
plt.show()

# ----------------------------------------------------------

# 5. Category-wise Average Demand
plt.figure(figsize=(10,5))
sns.barplot(data=df,
            x="Category",
            y="Demand")

plt.xticks(rotation=45)
plt.title("Average Demand by Category")
plt.show()

# ----------------------------------------------------------

# 6. Monthly Demand Trend
monthly_sales = df.groupby("Month")["Demand"].sum()

plt.figure(figsize=(10,5))
monthly_sales.plot(marker="o")

plt.title("Monthly Demand Trend")
plt.xlabel("Month")
plt.ylabel("Total Demand")
plt.grid(True)

plt.show()

# ----------------------------------------------------------

# 7. Promotion Effect on Demand
plt.figure(figsize=(6,5))
sns.boxplot(data=df,
            x="Promotion",
            y="Demand")

plt.title("Promotion vs Demand")
plt.show()

# ----------------------------------------------------------

# 8. Top 10 Selling Products
top_products = (
    df.groupby("Product_ID")["Demand"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

plt.figure(figsize=(10,5))
top_products.plot(kind="bar")

plt.title("Top 10 Selling Products")
plt.xlabel("Product ID")
plt.ylabel("Total Demand")
plt.show()

# ----------------------------------------------------------

# 9. Store-wise Average Demand
store_demand = df.groupby("Store_ID")["Demand"].mean()

plt.figure(figsize=(10,5))
store_demand.plot(kind="bar")

plt.title("Average Demand by Store")
plt.xlabel("Store ID")
plt.ylabel("Average Demand")
plt.show()

# ----------------------------------------------------------

# Save Clean Dataset
df.to_csv("retail_demand_cleaned.csv", index=False)

print("\n=======================================")
print("EDA Completed Successfully")
print("Cleaned dataset saved as retail_demand_cleaned.csv")
print("=======================================")
