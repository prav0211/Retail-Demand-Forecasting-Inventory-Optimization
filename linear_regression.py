import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title
st.title("📊 Retail Sales Prediction using Linear Regression")

# Load Dataset
df = pd.read_csv("retail_dataset.csv")

# Show Dataset
st.subheader("Dataset")
st.dataframe(df)

# Select Features
X = df[["Price", "Inventory"]]
y = df["Sales"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
st.subheader("Model Performance")

st.write("R² Score :", round(r2_score(y_test, y_pred),2))
st.write("MAE :", round(mean_absolute_error(y_test, y_pred),2))
st.write("RMSE :", round(mean_squared_error(y_test, y_pred)**0.5,2))

# Scatter Plot
st.subheader("Actual vs Predicted Sales")

fig, ax = plt.subplots()

ax.scatter(y_test, y_pred)

ax.set_xlabel("Actual Sales")
ax.set_ylabel("Predicted Sales")
ax.set_title("Linear Regression")

st.pyplot(fig)

# Prediction Section

st.subheader("Predict Sales")

price = st.number_input("Price", value=100)

inventory = st.number_input("Inventory", value=200)

if st.button("Predict"):

    prediction = model.predict([[price, inventory]])

    st.success(f"Predicted Sales : {prediction[0]:.2f}")





