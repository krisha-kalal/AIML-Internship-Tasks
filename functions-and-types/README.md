# Cost Function and It's Types, Loss Function and It's Types, and Optimizer and It's Types

## Cost Function

This function is a measure of the difference between predicted values and actual values. It is used to evaluate the accuracy of a model’s predictions and to guide the process of adjusting the model’s parameters in order to minimize the difference between predicted and actual values.

It is also known as an objective function.

A cost function measures the average loss over the entire dataset (or batch), often including additional penalty terms like regularization.

## Loss Function

A loss function is a mathematical formula used in machine learning and deep learning to evaluate how far the predicted output is from the actual target output.

It is also known as an error function. It helps improve the system performance.

It evaluates the prediction error on a single data point.

## Difference Between cost and loss function?

*The loss functions are defined on a single training example. It means it measures how well your model performing on a single training example. But if we consider the entire training set and try to measure how well is our model performing on it, we define a function called the cost function.*

*Because the cost function is built directly from the loss function, their underlying mathematical types are identical. You choose the type based on the problem type (Regression or Classification).*

## Types of cost function and loss function

### Regression Functions (for continuous outputs)

1. **MSE (Mean Squared Error):** This function calculates the average of the squared differences between the predicted values and the actual values. It penalizes larger errors more heavily.

2. **MAE (Mean Absolute Error):** This function calculates the absolute difference between actual and predicted values. It is less sensitive to large errors compared to MSE.

3. **Huber Loss:** This function behaves like MSE for small errors and like MAE for large errors. It is a good compromise if you have some outliers in your data.

### Classification Functions (for discrete outputs)

4. **Cross Entropy Loss:** This function combines a softmax layer and a negative log-likelihood loss in one step. It measures how well the predicted probabilities match the actual classes.

5. **Binary Cross-Entropy/Logloss:** This is a commonly used cost function for binary classification problems. It measures the difference between the predicted and actual probabilities of the binary output variable.

6. **Hinge Loss:** This function penalizes predictions that are not only incorrect but also far from the correct margin. It is used to maximize the margin between classes.

It is commonly used for training models for binary classification tasks using Support Vector Machines (SVMs).

### Probabilistic Function

7. **Kullback-Leibler Divergence Loss:** This function measures the difference between two probability distributions. It’s often used in tasks where you want to compare predicted distributions to actual ones.

## Optimizer

Optimizers are algorithms used in machine learning and deep learning to minimize the loss function (error) by adjusting model weights and biases during training.

### Weight

A weight is a multiplier applied to an input value. It tells the model how much attention to pay to a specific piece of information.

### Bias

A bias is a constant number added to the sum of the weighted inputs. It shifts the model’s prediction up or down, independent of the input features.

Without bias, if all the inputs are 0, the prediction would automatically be forced to be 0. Bias gives the model a starting baseline.

### Gradient

A gradient is a vector of partial derivatives that measures how much the loss value changes with respect to a small change in each individual weight and bias.

Because the gradient points uphill toward higher error, optimizers subtract the gradient to step downhill toward lower error.

### Landscapes of the loss function

The loss landscape is a continuous, multidimensional geometric terrain formed by plotting the value of the loss function against all possible combinations of the model's trainable parameters (its weights and biases).

1. **Global Minima:** The absolute lowest point on the loss function curve, representing the best possible solution.

2. **Local Minima:** A point lower than its neighbors but not the lowest possible point. Gradient Descent may get stuck here, leading to suboptimal solutions.

3. **Saddle Point:** A point where the gradient is zero but is neither a maximum nor a minimum. These can slow down convergence or cause the optimizer to stall.

### Types of optimizer

1. **Gradient Descent (GD)** 

- Gradient Descent is one of the earliest and most fundamental optimization techniques used in machine learning and neural networks. It forms the backbone of many modern optimization algorithms and remains an essential component of deep learning frameworks.

- The primary goal of Gradient Descent is to minimize the loss function by iteratively updating the model’s weights. This ensures that the network’s predictions become as accurate as possible by moving the weights toward the global minima — the point where the loss function achieves its lowest value.

2. **Stochastic Gradient Descent (SGD)**

- Unlike GD, which computes gradients using the entire dataset, SGD randomly selects a single sample or small batch from the dataset to update the weights during each iteration.

- This randomness introduces noise in the optimization path but offers distinct advantages in terms of speed and efficiency.

3. **Mini-Batch Gradient Descent**

- Mini-Batch Gradient Descent strikes a balance between Batch Gradient Descent (which uses the entire dataset) and Stochastic Gradient Descent (which uses only a single data point per update). Instead of relying on either extreme, this optimizer processes small batches of data at a time, making it both computationally efficient and relatively stable during training.

4. **Adagrad (Adaptive Gradient Descent) Optimizer**

- The defining feature of Adagrad is its ability to adapt the learning rate for each parameter during training based on the magnitude of past gradients. This adaptability makes it particularly effective for handling sparse datasets and features.

5. **RMSProp (Root Mean Square Propagation)**

- RMSProp addresses one of the key challenges in optimization — the variation in gradient magnitudes across parameters. In some cases, gradients may be small, while others may be disproportionately large, leading to inefficient updates when using a uniform learning rate for all weights.

- By maintaining a moving average of squared gradients and dividing the gradient by the square root of this average, RMSProp ensures that parameters with larger oscillations receive smaller updates.

6. **Adam (Adaptive Moment Estimation)**

- Combines the ideas of RMSprop (moving average of squared gradients / second moment) and SGD with Momentum (moving average of gradients / first moment). It is fast, handles noisy data well, and is widely considered the default go-to optimizer.

7. **Stochastic Gradient Descent with Momentum**

- Stochastic Gradient Descent (SGD) with Momentum is an advanced variant of the traditional SGD designed to address the inefficiencies caused by noisy and slow convergence.

- Momentum introduces a moving average of past gradients, smoothing out oscillations and pushing the optimizer toward a more stable path.

---

*Notes | Day 10 & Day 11*