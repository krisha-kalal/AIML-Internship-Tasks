# Ensemble Learning Methods, Boosting vs Bagging

## Ensemble Learning

In Machine Learning, a model can make two common types of errors:

- **High Bias (Underfitting):** The model is too simple to understand the patterns in the data.
- **High Variance (Overfitting):** The model learns the training data too closely, including unnecessary details and noise. As a result, it may perform poorly on new data.

**Ensemble Learning** is a technique that combines multiple machine learning models to improve overall prediction performance.

Instead of depending on a single model, it combines predictions from multiple models (called **base learners**) to produce a better result.

In simple words, multiple models work together to make better predictions.

### Types of ensemble learning methods

#### 1. Bagging (Bootstrap Aggregating)

Bagging trains multiple models independently using different random samples of the training data.

These samples are created using **Bootstrap Sampling**, which means selecting data points randomly with replacement. This allows the same data point to appear more than once in a sample.

After training, all models combine their predictions:
- **Classification:** Uses majority voting.
- **Regression:** Uses the average of predictions.

**Random Forest:** It is a popular Bagging algorithm that combines multiple decision trees. It also selects random subsets of features while building trees, which helps make the trees different from each other and improves predictions.

#### 2. Boosting

Boosting combines multiple weak learners to create a stronger model.

Unlike Bagging, models are trained one after another (sequentially). Each new model focuses on improving the mistakes made by the previous models.

In simple words, each new model tries to correct the errors of the models trained before it.

**Techniques of Boosting:**

- **AdaBoost (Adaptive Boosting):** It assigns higher weights to incorrectly predicted data points so that the next model focuses more on them. It is commonly used for classification.

- **XGBoost (Extreme Gradient Boosting):** An optimized version of Gradient Boosting. It uses regularization to control overfitting and improves training speed and performance.

- **Gradient Boosting:** It trains new models to predict the errors (residuals) made by previous models. It uses gradient descent to minimize the loss function and can be used for both classification and regression.

- **CatBoost (Categorical Boosting):** A Gradient Boosting algorithm that handles categorical features automatically, reducing the need for manual conversion of categories into numerical values.

#### 3. Stacking

Stacking combines predictions from different machine learning models using another model called a **Meta-Model**.

Each base model makes its own predictions, and the meta-model learns how to combine those predictions to produce a final result.

In simple words, multiple models make predictions, and one final model decides how to combine them.

*Note: The choice of ensemble method depends on factors such as prediction accuracy, training time, and model complexity.*

## Bagging vs Boosting

| Parameters | Bagging | Boosting |
|---|---|---|
| **Learning strategy** | Trains multiple models independently, often in parallel. | Trains models one after another (sequentially). |
| **Goal** | Mainly reduces variance and helps prevent overfitting. | Mainly reduces bias and improves weak models. |
| **Base Learners** | Usually uses strong models, such as deep decision trees. | Usually uses weak models, such as shallow decision trees. |
| **How data is used** | Uses different bootstrap samples of the training data. | Uses training data to focus on errors made by previous models. |
| **Interaction between models** | Models work independently. | Each new model learns from the previous models' mistakes. |
| **Popular algorithms** | Random Forest | AdaBoost, XGBoost, CatBoost, Gradient Boosting |
| **Parallelization** | Easier to parallelize because models are independent. | Harder to parallelize because models depend on previous ones. |
| **When to use** | When the model has high variance or overfits. | When the model has high bias or underfits. |
| **Overfitting behaviour** | Generally less prone to overfitting. | Can overfit if parameters are not tuned properly. |
| **How predictions are combined** | Uses majority voting for classification or averaging for regression. | Combines predictions from all learners, often using weighted contributions. |

---

*Notes | Day 16*