# Data Preprocessing, Feature Engineering and Model Evaluation

## Data Preprocessing

## Feature Selection

It is the process of choosing only the most useful input features for a machine learning model. It helps improve model performance, reduces noise and makes results easier to understand.

### Types of feature selection methods

1. **Filter Methods (The Quick Look Test):** These evaluate each feature independently with respect to the target variable. These methods are commonly used in the preprocessing phase to remove irrelevant or redundant features.

**Common Filter Techniques:**
- Information Gain: Measures reduction in entropy when a feature is used.
- Chi-square test: Checks the relationship between categorical features.
- Fisher’s Score: Ranks features based on class separability.
- Pearson’s Correlation Coefficient: Measures linear relationship between two continuous variables.
- Variance Threshold: Removes features with very low variance.
- Mean Absolute Difference: Similar to variance threshold but uses absolute differences.
- Dispersion ratio: Ratio of arithmetic mean to geometric mean; higher values indicate useful features.

2. **Wrapper Methods (The Trial and Error Test):** These evaluate different combinations of features by measuring their impact on model performance. They use search strategies to add or remove features and select the optimal subset based on predefined stopping criteria.

**Common Wrapper Techniques:**
- Forward Selection: Start with no features and add one at a time based on improvement.
- Backward Elimination: Start with all features and remove the least useful ones.
- Recursive Feature Elimination (RFE): Removes the least important features step by step.

3. **Embedded Methods (The Self Sorting Test):** These evaluate during the model training process. They combine the benefits of both filter and wrapper methods. Feature selection is integrated into the model training allowing the model to select the most relevant features based on the training process dynamically.

**Common Embedded Techniques:**
- L1 Regularization (Lasso): Keeps only features with non-zero coefficients.
- Decision Trees and Random Forests: Select features based on impurity reduction.
- Gradient Boosting: Pick features that reduce prediction error the most

## Model Evaluation

**Confusion Matrix:** It is a table that measures the performance evaluation of a classification model in machine learning. This helps you understand where the model is making mistakes so you can improve it. It breaks down the predictions into four categories:

| | Predicted Positive | Predicted Negative |
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN) |

- **True Positive (TP):** The model correctly predicted a positive outcome and the actual outcome was positive. eg: The model predicted win and the team won.

- **True Negative (TN):** The model correctly predicted a negative outcome and the actual outcome was negative. eg: The model predicted loose and the team lost.

- **False Positive (FP):** The model incorrectly predicted a positive outcome and the actual outcome was negative. It is also known as a Type I error. eg: The model predicted win and the team lost.

- **False Negative (FN):** The model incorrectly predicted a negative outcome and the actual outcome was positive. It is also known as a Type II error. eg: The model predicted loose and the team won.

---

*Notes | Day 5*