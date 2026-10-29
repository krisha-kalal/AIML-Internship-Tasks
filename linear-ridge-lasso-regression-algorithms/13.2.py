from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# load the Boston Housing dataset
diabetes= load_diabetes()
X, y = diabetes.data, diabetes.target

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize the features (important for regularization)
X_train /= X_train.std(axis=0)
X_test /= X_test.std(axis=0)

# initialize lasso and ridge models
lasso_model = Lasso(alpha=0.1)
ridge_model = Ridge(alpha=0.1)

# train the models
lasso_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

# make predictions on the test set
y_pred_lasso = lasso_model.predict(X_test)
y_pred_ridge = ridge_model.predict(X_test)

# calculate mean squared error and print the results
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

print("Lasso Mean Squared Error:", mse_lasso)
print("Ridge Mean Squared Error:", mse_ridge)