# Day 14 – Exploratory Data Analysis (EDA)

import pandas as pd

# Load dataset
df = pd.read_csv("students.csv")
print("First 5 rows of the dataset:")
print(df.head())

# Step 1: Basic Info and Shape
print("\n Dataset Shape and Info:")
print("Shape:", df.shape)
print(df.info())

# Step 2: Summary Statistics
print("\n Summary Statistics:")
print(df.describe())

# Step 3: Check for Missing Values
print("\n Missing Values:")
print(df.isnull().sum())

# Step 4: Categorical Value Counts
print("\n Gender Distribution:")
print(df["Gender"].value_counts())

print("\n City Distribution:")
print(df["City"].value_counts())

# Step 5: Correlation Analysis
print("\n Correlation Matrix:")
print(df.corr(numeric_only=True))

# Practice Exercise 1: Gender-wise Summary
print("\n Gender-wise Score Summary:")
summary = df.groupby("Gender")["Score"].describe()
print(summary)

# Practice Exercise 2: Average Score by City
print("\n Average Score by City:")
avg_by_city = df.groupby("City")["Score"].mean()
print(avg_by_city)

#  Practice Exercise 3: Top and Lowest Performer
print("\n Top Performer:")
top_student = df[df["Score"] == df["Score"].max()]
print(top_student)

print("\n Lowest Performer:")
low_student = df[df["Score"] == df["Score"].min()]
print(low_student)

# Practice Exercise 4: Detect Outliers Using IQR
print("\n Outliers (IQR Method):")
Q1 = df["Score"].quantile(0.25)
Q3 = df["Score"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["Score"] < (Q1 - 1.5 * IQR)) | (df["Score"] > (Q3 + 1.5 * IQR))]
print(outliers)

# Practice Exercise 5: Binning into Performance Levels
print("\n Binning Score into Performance:")
bins = [0, 60, 80, 100]
labels = ["Low", "Medium", "High"]
df["Performance"] = pd.cut(df["Score"], bins=bins, labels=labels)

print(df[["Name", "Score", "Performance"]])

print("End")

