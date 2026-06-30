import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")

df = pd.read_csv("retail_demand_dataset.csv")
print(df.head(5))

# Data Cleaning
# Remove Duplicate Records
df = df.drop_duplicates()

print(df.duplicated().sum())
#Fill Missing Numerical Values
numeric = df.select_dtypes(include=np.number)

for col in numeric.columns:
    df[col] = df[col].fillna(df[col].mean())
#Fill Missing Categorical Values
categorical = df.select_dtypes(include="object")

for col in categorical.columns:
    df[col] = df[col].fillna(df[col].mode()[0])
#Verify
print(df.isnull().sum())
#Convert Date Column
df["Date"] = pd.to_datetime(df["Date"])
#Extract Date Features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Weekday"] = df["Date"].dt.day_name()
print(df)