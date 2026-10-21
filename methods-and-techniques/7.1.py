# train test split method example
# import packages
import pandas as pd
from sklearn.model_selection import train_test_split

# creating a sample dataset
data = {
    'Feature_1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'Feature_2': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
    'Target_Label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

# importing data into a DataFrame
df = pd.DataFrame(data)

# head of the data
print(df.head(),"\n")

# Selecting features (X) and target (y) from the generated columns
X = df[['Feature_1', 'Feature_2']]
y = df['Target_Label']

# using the train test split function
X_train, X_test, y_train, y_test = train_test_split(X, y , random_state=42, test_size=0.2, shuffle=True)

# printing out train and test sets
print('X_train : ')
print(X_train.head())
print('\nX_test : ')
print(X_test.head())
print('\ny_train : ')
print(y_train.head())
print('\ny_test : ')
print(y_test.head())