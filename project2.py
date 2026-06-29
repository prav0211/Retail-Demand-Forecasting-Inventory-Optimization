import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")

df = pd.read_csv("retail_demand_dataset.csv")
print(df.head(5))
print(df.isnull().sum())
print(df.dtypes)
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())

