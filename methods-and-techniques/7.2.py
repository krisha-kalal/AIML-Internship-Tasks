# performing k-fold cross validation
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC # inbuilt support vector machine classifier
from sklearn.datasets import load_iris

# loading the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# creating an SVM classifier
svm_classifier = SVC()

# defining the number of folds for k-fold cross validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# performing cross validation
cross_val_results = cross_val_score(svm_classifier, X, y, cv=kf)

# printing the cross-validation results
print("Cross-Validation Results (Accuracy):")
for i, result in enumerate(cross_val_results, 1):
    print(f"  Fold {i}: {result * 100:.2f}%")

# printing the mean accuracy across all folds
print(f'Mean Accuracy: {cross_val_results.mean()* 100:.2f}%')