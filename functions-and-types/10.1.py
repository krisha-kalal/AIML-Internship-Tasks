# types of cost function and loss function implementation
import numpy as np

y_pred1 = np.array([1.2, 2.1, 3.5, 4.8]) # predicted values
y_actual1 = np.array([1, 2, 3, 5]) # actual values
n1 = len(y_pred1)

mse = 1/n1 * np.sum((y_pred1 - y_actual1)**2) # calculate mean squared error
print('MSE:', mse)

mae = 1/n1 * np.sum(np.abs(y_pred1 - y_actual1)) # calculate mean absolute error
print('MAE:', mae) 

y_pred2 = np.array([0.1, 0.5, 0.4])
y_actual2 = np.array([0.2, 0.3, 0.5])
kl = np.sum(y_actual2 * np.log(y_actual2 / y_pred2)) # calculate Kullback-Leibler divergence
print('Kullback-Leibler Divergence:', kl)

y_pred3 = np.array([0.2, 0.8, 0.6, 0.3])
y_actual3 = np.array([-1, 1, 1, -1])
hl = np.mean(np.maximum(0, 1 - y_actual3 * y_pred3)) # calculate hinge loss
print('Hinge Loss:', hl)

y_pred = np.array([0.2, 0.8, 0.6, 0.3])
y_actual = np.array([0, 1, 1, 0])
n2 = len(y_pred)
bce = -1/n2 * np.sum(y_actual * np.log(y_pred) + (1 - y_actual) * np.log(1 - y_pred)) # calculate binary cross-entropy
print('Binary Cross-Entropy:', bce)