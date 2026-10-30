import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# load dataset
iris = load_iris()
X = iris.data
y = iris.target

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = GaussianNB()
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# predict a new flower
prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
print("Predicted flower:", iris.target_names[prediction[0]])

# plot dataset
plt.scatter(X[:, 2], X[:, 3], c=y)
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Naive Bayes - Iris Flower Classification")
plt.show()