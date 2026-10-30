import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# create dataset
data = {
    "Study_Hours": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
    "Pass":        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["Study_Hours"]]
y = df["Pass"]

# train model
model = LogisticRegression()
model.fit(X, y)

# predict
prediction = model.predict(pd.DataFrame({"Study_Hours": [4.5]}))
print("Pass prediction for 4.5 study hours:", prediction[0])

# get probabilities for the plot
hours = pd.DataFrame({
    "Study_Hours": [i / 10 for i in range(10, 71)]
})
probabilities = model.predict_proba(hours)[:, 1]

# plot
plt.scatter(df["Study_Hours"], df["Pass"], label="Actual Results")
plt.plot(hours["Study_Hours"], probabilities, color="red",
         label="Predicted Probability")
plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression - Student Pass Prediction")
plt.legend()
plt.show()