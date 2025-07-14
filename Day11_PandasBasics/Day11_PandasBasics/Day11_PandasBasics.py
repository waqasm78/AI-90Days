# Day 11 - Pandas for Data Analysis

# Welcome Message
print("Welcome to Day 11 - Pandas for Data Analysis!")

# --- Importing Pandas ---
import pandas as pd

a = {1,2,3}
b = {2,3,4}
print(a-b)
# --- Creating Series and DataFrame ---
print("\nCreating Series and DataFrame")
series = pd.Series([10, 20, 30, 40])
print("Series:\n", series)

data = {
    "Name": ["Ali", "Sara", "John"],
    "Age": [25, 30, 22],
    "City": ["Lahore", "Karachi", "Islamabad"]
}
df = pd.DataFrame(data)
print("\nDataFrame:\n", df)

# --- Reading CSV File ---
print("\nReading CSV File")
df_csv = pd.read_csv("students.csv")
print(df_csv.head())

# --- Selecting Columns and Rows ---
print("\nSelecting Columns and Rows")
print("Names Column:\n", df_csv["Name"])
print("First Row:\n", df_csv.iloc[0])
print("Row 1 to 3:\n", df_csv.iloc[0:3])

# --- Filtering Rows ---
print("\nFiltering Rows")
adults = df_csv[df_csv["Age"] > 20]
print(adults)

# --- Basic Statistics ---
print("\nBasic Statistics")
print("Average Age:", df_csv["Age"].mean())
print("Max Age:", df_csv["Age"].max())

# --- Handling Missing Data ---
print("\nHandling Missing Data")
df_with_nan = pd.DataFrame({
    "Name": ["Ali", "Sara", None],
    "Score": [85, None, 90]
})
print("Original:\n", df_with_nan)
print("\nDrop Rows with NaN:\n", df_with_nan.dropna())
print("\nFill NaN with Value:\n", df_with_nan.fillna("Unknown"))

# --- Removing Duplicates ---
print("\nRemoving Duplicates")
df_dup = pd.DataFrame({
    "Name": ["Ali", "Sara", "Ali"],
    "Age": [25, 30, 25]
})
print("Original:\n", df_dup)
print("\nWithout Duplicates:\n", df_dup.drop_duplicates())

# --- Exporting to CSV ---
print("\nExporting to CSV")
df_csv.to_csv("students_cleaned.csv", index=False)
print("Data exported to 'students_cleaned.csv'")

# --- Practice Exercise 1 ---
print("\nPractice 1: Load and Filter")
data = pd.read_csv("students.csv")
females = data[data["Gender"] == "Female"]
print(females)

# --- Practice Exercise 2 ---
print("\nPractice 2: Add New Column")
data["Pass"] = data["Score"] >= 50
print(data.head())

# --- Practice Exercise 3 ---
print("\nPractice 3: Group by Gender")
grouped = data.groupby("Gender")["Score"].mean()
print(grouped)

