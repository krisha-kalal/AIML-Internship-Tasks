# Central Limit Theorem, Overfitting - Underfitting, Oversampling - Undersampling, Regression, Classification, Clustering

## 1. Sample Distribution

Sample distribution involves repeatedly taking samples from a population, calculating a statistic for each sample, and combining those values to create a distribution.

## 2. Central Limit Theorem (CLT)

The Central Limit Theorem states that if the sample size is sufficiently large, the sampling distribution of the sample mean will be approximately normal, even if the original population is not normally distributed, provided certain conditions are met.

A sample size of 30 or more is often used as a general guideline, but the required size depends on the data.

**Uses:**
- Helps understand the distribution of sample means.
- Helps estimate population values using sample data.
- Is useful in statistical analysis and hypothesis testing.

## 3. Overfitting and Underfitting

### Overfitting

Overfitting occurs when a model learns the training data too closely, including noise and unnecessary details. It performs well on training data but poorly on new, unseen data.

**Example:** A model achieves 99% accuracy on training data but only 65% accuracy on test data.

**How to reduce overfitting:**
- Use more training data.
- Apply regularization.
- Reduce model complexity.
- Use cross-validation.
- Apply pruning or dropout where appropriate.

### Underfitting

Underfitting occurs when a model is too simple to learn the important patterns in the data. It performs poorly on both training and test data.

**Example:** A model achieves 60% accuracy on training data and 58% accuracy on test data.

**How to reduce underfitting:**
- Use a more complex model.
- Add useful features.
- Reduce excessive regularization.
- Train the model appropriately.

## 4. Handling Imbalanced Data

Imbalanced data occurs when one class has significantly more observations than another class in a dataset. This can cause a machine learning model to favour the majority class and perform poorly on the minority class.

Two common techniques for handling imbalanced data are undersampling and oversampling.

### A. Undersampling

Undersampling reduces the number of observations in the majority class to make the classes more balanced.

**Techniques:**

1. **Random Undersampling:** Randomly removes observations from the majority class. It is simple but may remove useful information.

2. **Tomek Links:** Identifies pairs of nearby observations belonging to different classes. Removing majority-class observations from these pairs can help clarify the decision boundary.

### B. Oversampling

Oversampling increases the representation of the minority class to make the dataset more balanced.

**Techniques:**

1. **Random Oversampling:** Duplicates existing observations from the minority class. It is simple but may lead to overfitting because it adds no new information.

2. **SMOTE (Synthetic Minority Over-sampling Technique):** Creates synthetic observations by combining information from existing minority-class observations and their nearest neighbours.

3. **ADASYN (Adaptive Synthetic Sampling):** Creates more synthetic observations in areas where minority-class examples are harder to classify.

**Important:** Apply oversampling or undersampling only to the training data after splitting the dataset. This helps prevent data leakage.

## 5. Regression

Regression is a type of supervised machine learning used to predict continuous numerical values.

**Examples:**
- Predicting house prices.
- Predicting temperature.
- Estimating salary.
- Predicting sales.

### Regression Models

- **Linear Regression:** Predicts numerical values by learning a linear relationship between input features and the target.
- **Decision Tree Regressor:** Predicts numerical values using decision rules.
- **Random Forest Regressor:** Combines multiple decision trees to make numerical predictions.

## 6. Classification

Classification is a type of supervised machine learning used to predict categories or classes.

**Examples:**
- Predicting whether an email is spam or not.
- Predicting whether a student will pass or fail.
- Identifying whether a transaction is fraudulent.
- Classifying images into categories.

### Classification Models

- **Random Forest Classifier:** Combines multiple decision trees to predict a class.
- **KNN Classifier:** Predicts a class based on the classes of nearby data points.
- **Logistic Regression:** Predicts the probability of a class, commonly used for binary classification.
- **Decision Tree Classifier:** Uses decision rules to classify data. It can also be used as a regressor to predict data.
- **Support Vector Machine (SVM):** Finds a decision boundary that separates classes.

## 7. Clustering

Clustering is a type of unsupervised machine learning that groups similar data points together without using predefined target labels.

**Examples:**
- Grouping customers based on purchasing behaviour.
- Grouping products with similar characteristics.
- Identifying patterns in datasets.
- Grouping students based on study habits.

### Clustering Models

- **K-Means Clustering:** Divides data into a specified number of clusters based on similarity to cluster centres.
- **Hierarchical Clustering:** Creates a hierarchy of clusters by repeatedly combining or dividing groups.
- **Density-Based Clustering (DBSCAN):** Groups closely packed data points and identifies points in low-density areas as noise or outliers.

## 8. Difference Between Regression, Classification, and Clustering

| Regression | Classification | Clustering |
|---|---|---|
| Predicts continuous numerical values. | Predicts categories or classes. | Groups similar data points together. |
| Supervised learning. | Supervised learning. | Unsupervised learning. |
| Uses labelled training data. | Uses labelled training data. | Does not require labelled target data. |
| Example: Predicting house prices. | Example: Predicting pass or fail. | Example: Grouping customers. |

---

*Notes | Day 6*