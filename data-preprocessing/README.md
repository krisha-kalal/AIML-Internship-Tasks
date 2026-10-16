# Data Preprocessing, Feature Engineering and Model Evaluation

## Data Preprocessing

Data preprocessing is the process of cleaning and preparing raw data so that it can be easily used for analysis and machine learning.

### 1. Feature Scaling

Feature scaling is used to bring numerical features to a similar scale so that features with larger values do not unnecessarily influence certain machine learning models.

There are two main techniques:

#### A. Normalization

Normalization changes feature values to a range between 0 and 1.

**Formula:**

\[
x'=\frac{x-\min(x)}{\max(x)-\min(x)}
\]

#### B. Standardization

Standardization transforms data so that its mean becomes 0 and its standard deviation becomes 1. It is also known as Z-score standardization.

**Formula:**

\[
z=\frac{x-\mu}{\sigma}
\]

Where:
- \(x\) = Original value
- \(\mu\) = Mean
- \(\sigma\) = Standard deviation

### 2. Encoding Techniques

Encoding techniques convert categorical or non-numeric data into numerical values so that machine learning algorithms can process it.

#### Types of Encoding

1. **Label Encoding:** Assigns each category a unique integer. It is simple and memory-efficient but may create an unwanted order between categories.

2. **One-Hot Encoding:** Converts each category into a separate binary column. It avoids creating a false order but can create many columns when there are many categories.

3. **Target Encoding (Mean Encoding):** Replaces each category with the mean of the target variable for that category.

4. **Binary Encoding:** Converts categories into binary codes and stores the codes across multiple columns. It is useful for datasets with many unique categories but is slightly more complex to implement.

### 3. Imputation Techniques

Imputation is the process of filling missing values in a dataset instead of removing the entire row or column.

1. **Mean Imputation:** Replaces missing numerical values with the mean of the column.
2. **Median Imputation:** Replaces missing numerical values with the median of the column. It is useful when the data contains outliers.
3. **KNN Imputation:** Fills missing values using similar data points, known as nearest neighbors.
4. **Regression Imputation:** Uses a regression model to predict and fill missing numerical values.
5. **Mode Imputation:** Replaces missing values with the most frequently occurring value. It is commonly used for categorical data.

### 4. Handling Duplicates

Duplicate records are repeated rows in a dataset. They can affect the accuracy of analysis and should be identified and handled when necessary.

Common Pandas methods:

- `duplicated()`: Identifies duplicate rows.
- `drop_duplicates()`: Removes duplicate rows.

Duplicates can be handled by:
- Keeping the first occurrence.
- Keeping the last occurrence.
- Identifying duplicates based on specific columns.

### 5. Handling Outliers

Outliers are values that are very different from most other values in a dataset.

#### Identifying Outliers

1. **Visualization Methods:** Use box plots, scatter plots, and histograms to identify unusual values.
2. **Statistical Methods:** Use methods such as the Interquartile Range (IQR) and Z-score to detect outliers.

#### Handling Outliers

1. **Removal:** Remove outliers when they are errors or are not relevant to the analysis.
2. **Capping:** Limit extreme values to a specified upper or lower boundary.
3. **Mean/Median Imputation:** Replace unusual values with the mean or median when appropriate.

### 6. Handling Imbalanced Data

Imbalanced data refers to a situation where the number of instances across different classes in a classification problem is not evenly distributed. This can lead to biased models that favour the majority class and struggle to properly learn the characteristics of the minority class.

#### Techniques to Handle Imbalanced Data

1. **Oversampling:** Increases the number of samples in the minority class.
2. **Undersampling:** Reduces the number of samples in the majority class.
3. **Using appropriate evaluation metrics**

---

## Feature Selection

It is the process of choosing only the most useful input features for a machine learning model. It helps improve model performance, reduces noise and makes results easier to understand.

### Types of Feature Selection Methods

#### 1. Filter Methods (The Quick Look Test)

These evaluate each feature independently with respect to the target variable. These methods are commonly used in the preprocessing phase to remove irrelevant or redundant features.

**Common Filter Techniques:**

- **Information Gain:** Measures reduction in entropy when a feature is used.
- **Chi-square test:** Checks the relationship between categorical features.
- **Fisher’s Score:** Ranks features based on class separability.
- **Pearson’s Correlation Coefficient:** Measures linear relationship between two continuous variables.
- **Variance Threshold:** Removes features with very low variance.
- **Mean Absolute Difference:** Similar to variance threshold but uses absolute differences.
- **Dispersion ratio:** Ratio of arithmetic mean to geometric mean; higher values indicate useful features.

#### 2. Wrapper Methods (The Trial and Error Test)

These evaluate different combinations of features by measuring their impact on model performance. They use search strategies to add or remove features and select the optimal subset based on predefined stopping criteria.

**Common Wrapper Techniques:**

- **Forward Selection:** Start with no features and add one at a time based on improvement.
- **Backward Elimination:** Start with all features and remove the least useful ones.
- **Recursive Feature Elimination (RFE):** Removes the least important features step by step.

#### 3. Embedded Methods (The Self Sorting Test)

These evaluate during the model training process. They combine the benefits of both filter and wrapper methods. Feature selection is integrated into the model training allowing the model to select the most relevant features based on the training process dynamically.

**Common Embedded Techniques:**

- **L1 Regularization (Lasso):** Keeps only features with non-zero coefficients.
- **Decision Trees and Random Forests:** Select features based on impurity reduction.
- **Gradient Boosting:** Pick features that reduce prediction error the most.

---

## Model Evaluation

Model evaluation is the process of checking how well a machine learning model performs on data it has not seen before. It helps us understand whether the model makes accurate predictions.

### Confusion Matrix

It is a table that measures the performance evaluation of a classification model in machine learning. This helps you understand where the model is making mistakes so you can improve it. It breaks down the predictions into four categories:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN) |

- **True Positive (TP):** The model correctly predicted a positive outcome and the actual outcome was positive. eg: The model predicted win and the team won.

- **True Negative (TN):** The model correctly predicted a negative outcome and the actual outcome was negative. eg: The model predicted lose and the team lost.

- **False Positive (FP):** The model incorrectly predicted a positive outcome and the actual outcome was negative. It is also known as a Type I error. eg: The model predicted win and the team lost.

- **False Negative (FN):** The model incorrectly predicted a negative outcome and the actual outcome was positive. It is also known as a Type II error. eg: The model predicted lose and the team won.

---

*Notes | Day 5*