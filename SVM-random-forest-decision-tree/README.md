# SVM, Random Forest, Decision Tree

## SVM (Support Vector Machine)

Support Vector Machines (SVMs) are a type of supervised machine learning algorithm used for classification and regression tasks. SVMs work by finding the optimal hyperplane that separates data points into different classes.

**Hyperplane:** A hyperplane is a decision boundary that separates data points into different classes in a high-dimensional space. It can be used to make predictions on new data points by evaluating which side of the hyperplane they fall on. In an n-dimensional space, a hyperplane has (n-1)-dimensions.

**Margin:** A margin is the distance between the decision boundary (hyperplane) and the closest data points from each class. The goal of SVMs is to maximize this margin while minimizing classification errors. SVMs are designed to find the hyperplane that maximizes this margin, which is why they are sometimes referred to as maximum-margin classifiers.

**Support Vectors:** They are the data points that lie closest to the decision boundary (hyperplane) in a SVM. 

When a linear hyperplane is not possible, the input data is transformed into a higher-dimensional feature space, where it may be easier to find a linear decision boundary that separates the classes.

The **kernel trick** is a method used in SVMs to enable them to classify non-linear data using a linear classifier. By applying a kernel function, SVMs can implicitly map input data into a higher-dimensional space where a linear separator (hyperplane) can be used to divide the classes.

### Regression using SVM

SVMs, although generally used for classification can be used for both regression and classification. Support Vector Regression (SVR) is a machine learning algorithm used for regression analysis. It is different from traditional linear regression methods as it finds a hyperplane that best fits the data points in a continuous space, instead of fitting a line to the data points.

SVR in contrast to SVM tries to maximise the number of points in the street (margin), the width is controlled by a hyperparameter ε (epsilon).

## Decision Tree

Decision trees are one of the most intuitive and widely-used algorithms for classification and regression tasks. They model decision-making processes in a tree-like structure where each internal node represents a feature (attribute), each branch represents a decision rule, and each leaf node represents the outcome. 

Decision trees are easy to visualize and interpret, making them an excellent tool for explaining the logic behind predictions. It is a supervised learning algorithm. 

### How decision trees work?

1. **Selecting the Best Attribute:** Using a metric like Gini impurity, entropy, or information gain, the best attribute to split the data is selected.
2. **Splitting the Dataset:** The dataset is split into subsets based on the selected attribute.
3. **Repeating the Process:** The process is repeated recursively for each subset, creating a new internal node or leaf node until a stopping criterion is met.

To determine the best split, they rely on impurity metrics that evaluate how mixed a node’s class distribution is:

- **Entropy:** Entropy is often employed as a criterion to decide how to split data points at each node, aiming to create subsets that are more homogeneous with respect to the target variable.The feature and split point that result in the largest reduction in entropy are chosen for the split. The maximum entropy () occurs when instances are evenly distributed across all classes, creating a state of maximum disorder.

- **Gini Impurity:** A lower Gini impurity suggests a more homogeneous set of elements within the node, making it an attractive split in a decision tree. Decision tree algorithms aim to minimize the Gini impurity at each node, selecting the feature and split point that results in the lowest impurity.

## Random Forest

Random Forest is one of the most popular and powerful machine learning algorithms, used for both classification and regression tasks. It is an ensemble learning method. 

A random forest is essentially a collection of decision trees trained on different subsets of the data. Each tree in the forest is trained independently, and the final prediction is made by aggregating the predictions of all trees, either by taking the majority vote in classification or averaging the predictions in regression.

### Working of Random Forest

1. **Bootstrap Sampling (Bagging):** Creates multiple unique subsets of the original dataset with replacement. 
2. **Random Feature Selection:** Restricts each split in a tree to a random subset of features. 
3. **Grow Decision Trees:** Builds deep individual decision trees to their maximum depth without pruning.
4. **Aggregation:** Combines the predictions of all individual trees to produce a final output. 

### Hyperparameters of Random Forest

- n_estimators: Number of trees in the forest.
- max_depth: Maximum depth of each tree.
- max_features: Number of features considered for splitting at each node.
- criterion: Function used to measure split quality ('gini' or 'entropy').
- min_samples_split: Minimum samples required to split a node.
- min_samples_leaf: Minimum samples required to be at a leaf node.
- bootstrap: Whether to use bootstrap sampling when building trees (True or False).

---

*Notes | Day 15*