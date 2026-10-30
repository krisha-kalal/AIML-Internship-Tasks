import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# create dataset
data = {
    "Weight": [150, 160, 170, 180, 190, 200, 110, 120, 130, 140],
    "Size":   [7, 7.5, 8, 8.5, 9, 9.5, 5, 5.5, 6, 6.5],
    "Fruit":  ["Apple", "Apple", "Apple", "Apple", "Apple",
               "Apple", "Orange", "Orange", "Orange", "Orange"]
}

df = pd.DataFrame(data)

X = df[["Weight", "Size"]]
y = df["Fruit"]

# train KNN model
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=3)
)
model.fit(X, y)

# predict a new fruit
new_fruit = pd.DataFrame({"Weight": [155], "Size": [7.2]})
prediction = model.predict(new_fruit)
print("Predicted fruit:", prediction[0])

# plot dataset
for fruit in df["Fruit"].unique():
    subset = df[df["Fruit"] == fruit]
    plt.scatter(subset["Weight"], subset["Size"], label=fruit)

plt.scatter(155, 7.2, color="black", marker="X",
            s=150, label="New Fruit")

plt.xlabel("Weight (grams)")
plt.ylabel("Size (cm)")
plt.title("KNN - Fruit Classification")
plt.legend()
plt.show()