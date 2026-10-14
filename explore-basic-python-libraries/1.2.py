import numpy as np
import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Emma", "Bob"],
    "Department": ["HR", "IT", "IT", "HR", "Sales", "IT"],
    "Salary": [50000, 70000, np.nan, 52000, 65000, 75000],
}

# pd.DataFrame: Converts a given dictionary or dictionary of lists into a DataFrame.
df = pd.DataFrame(data)

# df.head: Displays the first 5 rows of the dataset.
print(df.head(3),"\n")

# df.info: Displays information about the DataFrame.
print(df.info(),"\n")

# df.drop_duplicates: Removes any rows with duplicate values.
df=df.drop_duplicates(subset=["Name"], keep="first")

# df.fillna(value): Replaces missing data values with a specified default or average.
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
print(df,"\n")

# df[df['column'] > value]: Filters rows based on a specific condition.
high_earners = df[df["Salary"] > 55000]
print(high_earners,"\n")

# df.groupby('column').mean(): Groups rows by a category and calculates averages for each group.
dept_averages = df.groupby("Department")["Salary"].mean()
print(dept_averages,"\n")

# df.dropna: Removes any rows containing missing or blank values.
clean_df = df.dropna()
print("Final clean data count:", len(clean_df))
