# Scikit-Learn and Data Cleaning

## 1. Scikit-Learn

**Scikit-Learn** is an open-source Python library used for Machine Learning. It provides various classification, regression, and clustering algorithms. A major part of Machine Learning logic and generating outputs is achieved with the help of this library, along with other specific libraries. It simplifies the process of building Machine Learning models.

### Installation and Import of Scikit-Learn

- **Installation:** `pip install scikit-learn`
- **Import:** `import sklearn`

### Some Submodules/Packages of Scikit-Learn

- **`sklearn.datasets`:** Loading toy datasets (e.g., Iris, Digits), fetching large real-world datasets from repositories (e.g., California Housing), or generating random shapes to test algorithms.

- **`sklearn.preprocessing`:** Scaling features (`StandardScaler`, `MinMaxScaler`), encoding categorical text (`OneHotEncoder`), or binarizing data.

- **`sklearn.ensemble`:** Combining multiple basic models to make a powerful "team" prediction. Examples: `RandomForestClassifier`, `GradientBoostingRegressor`.

- **`sklearn.model_selection`:** Splitting data (`train_test_split`), cross-validation testing, and automatic parameter search (`GridSearchCV`, `RandomizedSearchCV`).

- **`sklearn.metrics`:** Mathematical functions to judge performance. Classification tools (`accuracy_score`, `f1_score`, confusion matrices) and regression tools (`mean_squared_error`).

- **`sklearn.neighbors`:** Making predictions based on how physically close data points are to each other. Examples: `KNeighborsClassifier`, `RadiusNeighbors`.

## 2. Workflow of Scikit-Learn

### 1. Loading a Dataset

A dataset consists of:

- **Features (X):** Input variables that describe the data.
- **Target (y):** The value we want to predict.

Scikit-Learn provides built-in datasets like Iris, Digits, and Boston Housing.

### 2. Preprocessing and Feature Engineering

- **Preprocessing (Cleaning):** We handle missing values and fix formatting issues.
- **Feature Engineering (Enhancement):** We create or modify features (columns) to help the model learn better. For example, use Label Encoding or One-Hot Encoding instead of Yes/No, using `0` and `1` because the model predicts with binary numbers better.
- **Scaling (Normalization):** If the data has data points far from each other, scaling is a technique to make them closer to each other.

### 3. Splitting the Data

- **Split:** The data is split into training and testing datasets.
- **Train:** The model is trained on 70%–80% of the data.
- **Test:** This checks how well the model performs on unseen data. Typically, 20%–30% is used.

### 4. Model Selection and Training

We choose the appropriate Machine Learning algorithm based on the given task. For example, we can use Linear Regression when we want to predict numerical data. During training, we fit the algorithm to the training set using the `.fit()` method in Python.

### 5. Hyperparameter Tuning

Algorithms come with default configuration settings called **hyperparameters** that dictate how the model learns. For example, in a Decision Tree model, a major hyperparameter is `max_depth` (how deep the tree can grow). It is used to optimize the performance of a model.

### 6. Model Prediction and Evaluation

With the final tuned model ready, we give it the feature values from the testing set. The model outputs its predictions using the `.predict()` method.

## 3. Data Cleaning Techniques

Data cleaning is required before giving a dataset to a model for prediction or identifying patterns. If the data is not cleaned properly, the model may not predict accurately. The Pandas and NumPy libraries play a major role in cleaning the data.

### Common Data Cleaning Techniques

| Task | Functions Used |
|---|---|
| Find missing values | `isna()`, `isnull()` |
| Delete null/missing values | `dropna()` |
| Delete duplicates | `drop_duplicates()` |
| Fill out missing values (using imputation: mean, median, or mode) | `fillna()` |
| Replace values (numbers, strings, booleans) | `replace()`, `map()` |
| Change data types | `pd.to_numeric()`, `astype()` |
| Clean text columns | `str.strip()`, `str.lower()`, `str.replace()` |
| Handle outliers | `clip()` |

---

*Notes | Day 2*