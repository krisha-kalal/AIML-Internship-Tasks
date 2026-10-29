# Linear Regression, Ridge Regression and Lasso Regression

## Linear Regression

Linear regression is a statistical technique used to model the linear relationship between a dependent variable and one or more independent variables.

The dependent variable is the variable that is being predicted, while the independent variables are the variables that are used to predict the dependent variable. Linear regression is used to find the equation of a line that best fits a set of data points.

It is one of the most basic types of regression in supervised machine learning.

The line of best fit is considered to be the line for which the error between the predicted values and the observed values is minimum.

It is also called the regression line and the errors are also known as residuals.

### Equation of Linear Regression

The linear regression hypothesis states that there is a linear relationship between a dependent variable (Y) and one or more independent variables (X). This means that as the value of X changes, the value of Y is also expected to change in a linear fashion.

**Simple Linear Regression (One Predictor):**

$$
y = b_0 + b_1x
$$

**(or y = mx + b)**

**Multiple Linear Regression (Multiple Predictors):**

$$
y = b_0 + b_1x_1 + b_2x_2 + \dots + b_nx_n
$$

- **y:** The predicted output (dependent variable).
- **b₀:** The intercept (or model bias).
- **b₁, b₂, ...:** The feature weights or slopes.
- **x₁, x₂, ...:** The input features (independent variables).

### Metrics Used to Calculate Errors in the Model

Some of the metric tools that can be used to calculate errors in the model are:

1. **MSE (Mean Squared Error)**
2. **RMSE (Root Mean Squared Error)**
3. **MAE (Mean Absolute Error)**
4. **MAPE (Mean Absolute Percentage Error)**
5. **R² (R-Squared)**

## Ridge Regression and Lasso Regression

In the field of machine learning, linear regression is a powerful tool for predicting outcomes based on input features.

But when dealing with datasets that have a large number of features, linear regression can become prone to overfitting. Overfitting occurs when the model fits the training data too closely, capturing noise and leading to poor generalization on new data. To solve this challenge, regularization techniques like Lasso and Ridge regression come in.

Ridge Regression and Lasso Regression are both regularization techniques used in linear regression to prevent overfitting by adding a penalty term to the cost and loss functions.

### Ridge Regression (L2 Regularization)

It adds a penalty proportional to the sum of the squares of the coefficients.

**Equation:**

$$
\text{Cost} = \text{MSE} + \lambda \sum_{j=1}^{n} b_j^2
$$

- **MSE:** Mean Squared Error.
- **λ (Lambda):** Controls the strength of the penalty.
- **bⱼ:** Model coefficients.

### Lasso Regression (L1 Regularization)

It adds a penalty proportional to the sum of the absolute values of the coefficients.

**Equation:**

$$
\text{Cost} = \text{MSE} + \lambda \sum_{j=1}^{n} |b_j|
$$

- **MSE:** Mean Squared Error.
- **λ (Lambda):** Controls the strength of the penalty.
- **bⱼ:** Model coefficients.

### Choosing Between Lasso and Ridge

The choice between Lasso and Ridge depends on the specific characteristics of the dataset and the goals of the modeling task:

- **Lasso:** Preferable when feature selection is crucial and a sparse model is desired.
- **Ridge:** Suitable when all features should be retained, and multicollinearity is a concern.

### Key difference between Ridge and Lasso

- **Ridge Regression:** Shrinks the coefficients towards zero but usually does not make them exactly zero.
- **Lasso Regression:** Can reduce some coefficients to exactly zero, which helps in feature selection.

---

*Notes | Day 13*