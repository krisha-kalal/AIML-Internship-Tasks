import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

# generate synthetic data
np.random.seed(0)
x = 2 * np.random.rand(100,1)
y = 4 + 3 * x + np.random.randn(100,1)
# Y = 4 + 3x + noise
# y = mx + b

# plot the synthetic data
plt.scatter(x,y)
plt.title('Scatter plot of the synthetic data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# set up and train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# make predictions on the test set
y_pred = model.predict(x_test)

# calculate mean squared error and r-squared score
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

# print the results
print(f'Coefficient: {model.coef_[0][0]}')
print(f'Intercept: {model.intercept_[0]}')
print(f'Mean Squared Error: {mse}')

# plot the points
plt.scatter(x_test, y_test, color='black', label='Actual data')
plt.plot(x_test, y_pred, color='blue', linewidth=3, label='Regression line')
plt.title('Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()