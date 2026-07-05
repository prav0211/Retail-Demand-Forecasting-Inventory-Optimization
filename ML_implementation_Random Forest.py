import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("retail_demand_dataset.csv")

# Convert Date
df["Date"] = pd.to_datetime(df["Date"])

# Create Date Features
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# Encode Categorical Columns
le = LabelEncoder()

df["Store"] = le.fit_transform(df["Store"])
df["Department"] = le.fit_transform(df["Department"])
df["Category"] = le.fit_transform(df["Category"])

# Features
X = df[[
    "Store",
    "Department",
    "Category",
    "Sales",
    "Inventory",
    "Price",
    "Promotion",
    "Holiday",
    "Month",
    "Day",
    "Reorder_Level"
]]

# Target
y = df["Demand_Forecast"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

# Evaluation
print("Mean Absolute Error :", mean_absolute_error(y_test, prediction))
print("Root Mean Squared Error :", np.sqrt(mean_squared_error(y_test, prediction)))
print("R2 Score :", r2_score(y_test, prediction))
