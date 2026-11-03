from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

data = load_wine()

x = data.data # features
y = data.target # labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# bagging with random forest
from sklearn.ensemble import RandomForestClassifier

bagging_model = RandomForestClassifier(n_estimators=100, random_state=42) # here the model creates 100 decision trees and makes a prediction based on the majority vote of the trees
bagging_model.fit(x_train, y_train)

print("Bagging accuracy:", bagging_model.score(x_test, y_test))

# boosting with gradient boosting
from sklearn.ensemble import GradientBoostingClassifier

boosting_model = GradientBoostingClassifier(n_estimators=100, random_state=42) # here the model creates 100 decision trees and makes a prediction based on the weighted average of the trees
boosting_model.fit(x_train, y_train)

print("Boosting accuracy:", boosting_model.score(x_test, y_test))

# stacking with logistic regression and decision tree
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

base_models = [
    ('decision_tree', DecisionTreeClassifier()),
    ('logistic_regression', LogisticRegression())
]

stacking_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression()) # here the model creates a meta-model that combines the predictions of the base models using logistic regression
stacking_model.fit(x_train, y_train)

print("Stacking accuracy:", stacking_model.score(x_test, y_test))