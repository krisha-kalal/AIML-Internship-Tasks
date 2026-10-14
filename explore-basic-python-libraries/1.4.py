import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

raw_data = {
    "Age": [25, 47, np.nan, 35, 52],
    "Bought": [0, 1, 0, 1, 1] 
}
df = pd.DataFrame(raw_data)

df["Age"] = df["Age"].fillna(df["Age"].mean())
print("Cleaned Pandas DataFrame")
print(df, "\n")

X = df["Age"].to_numpy()

X_scaled = X / X.max()
print("Normalized NumPy Array for ML Input")
print(X_scaled, "\n")

plt.figure(figsize=(5, 3))
plt.scatter(df["Age"], df["Bought"], color="red", marker="X", s=100)
plt.title("ML Exploratory Plot: Age vs Buying Behavior")
plt.xlabel("Age")
plt.ylabel("Bought (0 = No, 1 = Yes)")
plt.grid(True, linestyle="--")
plt.show()
