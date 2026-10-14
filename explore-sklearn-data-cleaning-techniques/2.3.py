import pandas as pd
import numpy as np

data = {
    "Name": ["Alice", "Bob    ", "Charlie", "David", "Emma", "Bob", "Alice"],
    "Department": ["HR", "IT", "it", "hr", "Sales", "Sles", "HR"],
    "Salary": [50000, 70000, np.nan, "52000", np.nan, 75000, 50000],
    "Age": [25, 30, 35, 999, 28, -5, 25],
}

df = pd.DataFrame(data)
print("Original Data:",df,"\n")

df["Name"] = df["Name"].str.strip() # remove leading/trailing spaces

print("Missing Values:",df.isnull().sum()) # finds total missing values in each column

df = df.drop_duplicates(subset=["Name", "Department"], keep="last") # drop duplicates

df["Department"] = df["Department"].replace({"Sles": "sales"}) # correct typos
df["Department"] = df["Department"].str.title() # standardize department names

df["Salary"] = pd.to_numeric(df["Salary"], errors='coerce') # convert salary to numeric
df["Salary"] = df["Salary"].fillna(df["Salary"].mean()) # fill missing salary with mean

df["Age"] = df["Age"].clip(lower=10, upper=90) # handle outliers in age
df["Age"] = df["Age"].fillna(df["Age"].mean()) # fill missing age with mean

print("Cleaned Data:",df)