import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
df = pd.read_csv("retail_demand_dataset.csv")
print(df.head())
print(df.info())
print(df.describe())

# convert date
df["Date"] = pd.to_datetime(df["Date"])

# prophet need only 2 parameters
prophet_df = df[["Date","Demand_Forecast"]]
prophet_df.columns = ["ds","y"]
print(prophet_df.head())

model = Prophet()
model.fit(prophet_df)
future = model.make_future_dataframe(
    periods=30
)


# Forcasting
forecast = model.predict(future)
print(
    forecast[
        ["ds","yhat","yhat_lower","yhat_upper"]
    ].tail()
)

# Forcasting Graph
fig = model.plot(forecast)
plt.title("Retail Demand Forecast")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.show()

# Trend component
model.plot_components(forecast)
plt.show()

forecast.to_csv(
    "forecast_output.csv",
    index=False
)

print("Forecast saved successfully.")

# forecast_output.csv >> This will be the output file.
