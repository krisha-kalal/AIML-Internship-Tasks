# importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# load data
data = {'Exam1_Score':[78,85,60,72,90,88,68,55,80,91],
        'Exam2_Score':[82,90,50,78,85,92,70,65,88,95],
        'Admitted':[1,1,0,1,1,1,0,0,1,1] # 1 means admitted and 0 means rejected 
    }

df = pd.DataFrame(data)
print(df)

# splitting data into features x and target y
x = df[['Exam1_Score','Exam2_Score']]
y = df['Admitted']

# splits data so 80% is used for training the model, and 20% (2 out of the 10 students) is saved to quiz the model later (test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# model selection
model = LogisticRegression()
# model looks at x_train (scores) and y_train (results) to calculate mathematical boundary line between who gets admitted and rejected
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# performance evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy:{accuracy}')

print('Classification Report:\n',classification_report(y_test, y_pred))

labels = [0,1]
conf_matrix = confusion_matrix(y_test, y_pred, labels = labels)
print('Confusion Matrix\n',conf_matrix)
