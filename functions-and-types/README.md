# Cost Function and It's Types, Loss Function and It's Types, and Optimizer and It's Types

## Cost Function

This function is a measure of the difference between predicted values and actual values. It is used to evaluate the accuracy of a model’s predictions and to guide the process of adjusting the model’s parameters in order to minimize the difference between predicted and actual values.

It is also known as an objective function.

A cost function measures the average loss over the entire dataset (or batch), often including additional penalty terms like regularization.

## Loss Function

A loss function is a mathematical formula used in machine learning and deep learning to evaluate how far the predicted output is from the actual target output.

It is also known as an error function. It helps improve the system performance.

It evaluates the prediction error on a single data point.

## Difference Between Cost and Loss Function?

*The loss functions are defined on a single training example. It means it measures how well your model performing on a single training example. But if we consider the entire training set and try to measure how well is our model performing on it, we define a function called the cost function.*

*Because the cost function is built directly from the loss function, their underlying mathematical types are identical. You choose the type based on the problem type (Regression or Classification).*

## Types of Cost Function and Loss Function

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

1. **Kullback-Leibler Divergence Loss:** This function measures the difference between two probability distributions. It’s often used in tasks where you want to compare predicted distributions to actual ones.

---

*Notes | Day 10*