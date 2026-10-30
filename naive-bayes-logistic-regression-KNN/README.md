# Classification Algorithms: Naive Bayes, Logistic Regression and KNN (K-Nearest Neighbors)

## Naive Bayes

Naive Bayes is a supervised machine learning algorithm used for classification. It predicts the class of new data by calculating the probability of each possible class.

It is based on **Bayes' Theorem**, which explains how to update the probability of an event when new information becomes available.

### Bayes' Theorem

Bayes' Theorem states that the probability of an event occurring can be calculated using prior knowledge of related conditions.

**Formula:**

$$
P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}
$$

where:
- **P(A|B):** Probability that the data belongs to class A given the features B.
- **P(B|A):** Probability of observing the features B given class A.
- **P(A):** Prior probability of class A.
- **P(B):** Probability of observing the features B.

The algorithm calculates the probability of each class and predicts the class with the highest probability.

It is called *naive* because it assumes that all features are conditionally independent of one another given the class.

For example, when classifying an email as spam or not spam, the algorithm considers the words in the email as features and calculates how likely they are to appear in each class.

### Working of Naive Bayes

1. **Training:** The model learns the probabilities of different classes and their features from labeled data.
2. **Prediction:** For new data, it calculates the probability of each class using Bayes' Theorem.
3. **Classification:** It predicts the class with the highest probability.

## Logistic Regression

Logistic regression is a supervised machine learning algorithm mainly used for classification tasks. It predicts the probability that an instance belongs to a particular class.

Despite its name, it is mainly used for classification rather than predicting continuous numerical values.

It uses the **sigmoid function** to convert the model's output into a probability between 0 and 1.

### Hypothesis of Logistic Regression

$$
p(y = 1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + \beta_2x_2)}}
$$

where:
- **p(y = 1):** Probability that the outcome belongs to class 1.
- **e:** Euler's number, approximately 2.718.
- **β₀:** Intercept.
- **β₁, β₂:** Coefficients of the predictor variables.
- **x₁, x₂:** Input features.

The sigmoid function produces a probability between 0 and 1. A classification threshold, commonly 0.5, can then be used to assign a class.

## KNN (K-Nearest Neighbors)

KNN is a supervised machine learning algorithm used for classification. It classifies new data points by comparing them with the nearest data points in the training dataset.

For example, suppose we have a dataset containing tomatoes and bananas. The model uses features such as shape and color to compare a new object with existing examples and predict its class.

### Working of KNN

1. Choose the value of **K**, which represents the number of nearest neighbors to consider.
2. Calculate the distance between the new data point and the existing data points.
3. Identify the K nearest neighbors.
4. Count the classes of those neighbors.
5. Assign the class that appears most frequently among the neighbors.

Choosing the right value of K is called parameter tuning and is necessary for better results. A common starting heuristic is to try the square root of the number of training data points, but the best value should be selected through validation.

### Euclidean Distance Formula

Euclidean distance is commonly used to calculate the distance between two data points.

$$
d = \sqrt{(x-a)^2 + (y-b)^2}
$$

where:
- **d:** Distance between the two points.
- **(x, y):** Coordinates of the first point.
- **(a, b):** Coordinates of the second point.

KNN identifies the nearest neighbors by comparing their distances. For classification, it usually assigns the majority class among those neighbors.

---

*Notes | Day 14*