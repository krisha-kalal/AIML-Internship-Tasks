# NumPy, Pandas, and Matplotlib

## NumPy
NumPy: It is a python library which is used for scientific computing and mathematical operations. It is also used for creating arrays and performing various array operations. These arrays are specialized homogenous so we can create an array of any single datatype only. NumPy stands for Numerical Python.

## Pandas
Pandas: It is also a python library mainly used to analyse, clean, explore, and manipulating datasets. This library is very useful for Data Science and Machine Learning. Statistical analysis is also done using pandas. Pandas stands for Panel Data or Python Data Analysis.

## Matplotlib
Matplotlib: It is a python library used for static, animated, and interactive data visualizations which helps the users and clients to understand the data, trends, patterns visually instead of just the numbers.

## Real World Use-Case of NumPy, Pandas and Matplotlib in the field of Machine Learning

### 1. Pandas: Preparing the Dataset

Before training a machine learning model, your data must look like a clean spreadsheet. It also helps in feature engineering.

**Use Case:** Handling missing values and encoding categories. If a dataset has blank spaces (e.g., a missing house price) or text (e.g., "New York"), an ML algorithm will crash.

**Example:** Using Pandas to instantly fill missing age values with the average age, and converting text categories like Yes/No into numbers 1 or 0.

### 2. NumPy: Feeding the Math Engine

Machine learning models are essentially massive mathematical equations that require data to be in structured blocks of numbers. It helps in feature engineering and normalization.

**Use Case:** Splitting data into Features X and Labels y, and Normalizing Features. Models learn better when all numerical inputs are on the same small scale (like between 0 and 1). If one input is "Salary" ($100,000) and another is "Age" (25), the model will mistakenly think Salary is 4000 times more important just because the number is bigger.

**Example:** Converting a Pandas table into a NumPy array and dividing all values by the maximum value to scale them down perfectly.

### 3. Matplotlib: Spotting Patterns & Errors

You cannot build a good ML model blindly. You must see what your data looks like before and after training.

**Use Case:** Plotting feature correlations and tracking model loss. Matplotlib helps you visually check if your data has outliers before training. After training, it helps you plot a line chart of the model's accuracy over time to see if it is actually improving.

**Example:** Plotting a scatter plot of "House Size" vs "Price" to confirm there is a linear pattern for a Linear Regression model to learn.

---

*Notes | Day 1*