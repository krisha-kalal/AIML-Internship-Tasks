import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsRegressor
X, y = load_iris(return_X_y=True)
print(X, y)
Model = KNeighborsRegressor()
Model.fit(X, y)
pred = Model.predict(X)
print(pred)
plt.scatter(pred,y)
plt.show()