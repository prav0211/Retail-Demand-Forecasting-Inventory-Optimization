# ==========================================================
# Week 3 - Day 4 to Day 6
# LightGBM Demand Forecasting
# ==========================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from lightgbm import LGBMRegressor

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("retail_demand_dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())

# ==========================================================
# Convert Date
# ==========================================================

df["Date"] = pd.to_datetime(df["Date"])

# ==========================================================
# Feature Engineering
# ==========================================================

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# ==========================================================
# Encode Categorical Columns
# ==========================================================

encoder = LabelEncoder()

df["Store"] = encoder.fit_transform(df["Store"])

df["Department"] = encoder.fit_transform(df["Department"])

df["Category"] = encoder.fit_transform(df["Category"])

# Product_ID is a string, so encode it too
df["Product_ID"] = encoder.fit_transform(df["Product_ID"])

# ==========================================================
# Features
# ==========================================================

X = df[
    [
        "Store",
        "Department",
        "Category",
        "Product_ID",
        "Sales",
        "Inventory",
        "Price",
        "Promotion",
        "Holiday",
        "Reorder_Level",
        "Year",
        "Month",
        "Day"
    ]
]

# ==========================================================
# Target Variable
# ==========================================================

y = df["Demand_Forecast"]

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Samples :", len(X_train))
print("Testing Samples :", len(X_test))

# ==========================================================
# Build LightGBM Model
# ==========================================================

model = LGBMRegressor(

    n_estimators=200,

    learning_rate=0.05,

    max_depth=8,

    random_state=42

)

# ==========================================================
# Train Model
# ==========================================================

model.fit(X_train, y_train)

print("Model Training Completed")

# ==========================================================
# Prediction
# ==========================================================

prediction = model.predict(X_test)

# ==========================================================
# Evaluation
# ==========================================================

mae = mean_absolute_error(y_test, prediction)

rmse = np.sqrt(mean_squared_error(y_test, prediction))

r2 = r2_score(y_test, prediction)

print("\nModel Performance")
print("----------------------------")
print("MAE :", round(mae,2))
print("RMSE :", round(rmse,2))
print("R2 Score :", round(r2,2))

# ==========================================================
# Actual vs Predicted
# ==========================================================

result = pd.DataFrame({

    "Actual_Demand": y_test,

    "Predicted_Demand": prediction

})

print(result.head())

# ==========================================================
# Save Prediction
# ==========================================================

result.to_csv(

    "predicted_vs_actual.csv",

    index=False

)

print("Prediction file saved successfully")

# ==========================================================
# Feature Importance
# ==========================================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nFeature Importance")

print(importance)

# ==========================================================
# Plot Feature Importance
# ==========================================================

plt.figure(figsize=(10,6))

plt.bar(

    importance["Feature"],

    importance["Importance"]

)

plt.xticks(rotation=45)

plt.title("LightGBM Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.tight_layout()

plt.show()

# ==========================================================
# Actual vs Predicted Plot
# ==========================================================

plt.figure(figsize=(8,6))

plt.scatter(

    y_test,

    prediction

)

plt.xlabel("Actual Demand")

plt.ylabel("Predicted Demand")

plt.title("Actual vs Predicted Demand")

plt.show()

# ==========================================================
# Residual Plot
# ==========================================================

residual = y_test - prediction

plt.figure(figsize=(8,6))

plt.scatter(

    prediction,

    residual

)

plt.axhline(

    y=0,

    color="red"

)

plt.xlabel("Predicted Demand")

plt.ylabel("Residual Error")

plt.title("Residual Plot")

plt.show()

print("\nLightGBM Model Completed Successfully")

# After running the code, you will get: predicted_vs_actual.csv
