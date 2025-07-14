# Day 11 - Pandas for Data Analysis 🐼📊

Welcome to **Day 11** of your AI-90Days journey! Today, you'll explore **Pandas**, a powerful Python library for working with structured data. Pandas makes it easy to clean, transform, filter, and analyze datasets, especially those in CSV or Excel format.

---

## 🌟 Objectives

* Learn what Series and DataFrames are
* Load CSV data into a DataFrame
* Select and filter rows and columns
* Perform basic statistics
* Handle missing values and duplicates
* Export cleaned data to CSV
* Practice with real-world examples

---

## 🔍 What is Pandas?

**Pandas** is a Python library built on top of NumPy. It provides fast, flexible, and expressive data structures like **Series** and **DataFrame** to work with structured data efficiently. It is widely used in data science and machine learning pipelines for data manipulation.

---

## 🔧 Setup

Make sure you have Pandas installed:

```bash
pip install pandas
```

You can test this in **Visual Studio 2022** or **Jupyter Notebook**. Create a project called `Day11_Pandas_Basics`.

---

## 📃 Creating Series and DataFrame

Pandas offers two core data structures: `Series` and `DataFrame`. A **Series** is a one-dimensional array-like object with labels, and a **DataFrame** is a two-dimensional table of data with rows and columns, like an Excel sheet.

```python
import pandas as pd

series = pd.Series([10, 20, 30, 40])
print(series)

data = {
    "Name": ["Ali", "Sara", "John"],
    "Age": [25, 30, 22],
    "City": ["Lahore", "Karachi", "Islamabad"]
}
df = pd.DataFrame(data)
print(df)
```

**Explanation:** You can store and view tabular data similar to an Excel sheet.

---

## 📄 Reading CSV Files

Often, data is stored in external files like `.csv`. Pandas makes it very easy to read and load such data directly into a DataFrame using the `read_csv()` function.

```python
df = pd.read_csv("students.csv")
print(df.head())
```

**Explanation:** `read_csv()` reads your file, and `head()` shows the first 5 rows.

---

## 🔢 Selecting Columns and Rows

Once data is loaded into a DataFrame, you can easily access specific columns or rows using indexing. This allows you to extract relevant portions of your data for analysis.

```python
print(df["Name"])        # Select a column
print(df.iloc[0])         # Select first row
print(df.iloc[0:3])       # Select first 3 rows
```

**Explanation:** Use `[]` for columns, and `iloc` for row indexing.

---

## 📅 Filtering Rows

Filtering helps you narrow down data based on specific conditions. For example, selecting rows where Age is greater than 20.

```python
adults = df[df["Age"] > 20]
print(adults)
```

**Explanation:** Filters rows where the condition is true.

---

## 📊 Basic Statistics

Pandas allows quick summary statistics like mean, max, min, and more, directly on numeric columns. This is useful for getting a quick overview of your data.

```python
print("Mean:", df["Age"].mean())
print("Max:", df["Age"].max())
```

**Explanation:** You can quickly compute statistics like mean, max, min, etc.

---

## 🚫 Handling Missing Data

Real-world data often contains missing values. Pandas offers functions to drop or fill these missing values to keep your dataset clean and usable.

```python
df_with_nan = pd.DataFrame({
    "Name": ["Ali", "Sara", None],
    "Score": [85, None, 90]
})

print(df_with_nan.dropna())
print(df_with_nan.fillna("Unknown"))
```

**Explanation:** `dropna()` removes rows with missing data. `fillna()` replaces them.

---

## ❌ Removing Duplicates

Datasets may have duplicate rows that can affect analysis. Use `drop_duplicates()` to remove repeated entries.

```python
df_dup = pd.DataFrame({
    "Name": ["Ali", "Sara", "Ali"],
    "Age": [25, 30, 25]
})
print(df_dup.drop_duplicates())
```

**Explanation:** Useful for cleaning datasets with repeated entries.

---

## 📂 Exporting to CSV

After cleaning and transforming data, you might want to save it back to disk. Use `to_csv()` to export your DataFrame to a CSV file.

```python
df.to_csv("students_cleaned.csv", index=False)
```

**Explanation:** Saves your cleaned DataFrame to a new CSV file.

---

## 💡 Practice Exercises

### ✅ Exercise 1: Filter Female Students

Let's filter and display only the rows where the student's gender is "Female". This helps in understanding how category-based filtering works in real-world data.

```python
data = pd.read_csv("students.csv")
females = data[data["Gender"] == "Female"]
print(females)
```

**Goal:** Learn to filter based on category (like gender).

---

### ✅ Exercise 2: Add a New Column

You can add new columns to a DataFrame based on conditions or calculations. Let's create a new column that shows whether each student passed (score >= 50).

```python
data["Pass"] = data["Score"] >= 50
print(data.head())
```

**Goal:** Add a derived column based on conditions.

---

### ✅ Exercise 3: Group by Gender and Calculate Average Score

Grouping data helps summarize and compare categories. This example shows how to calculate average score by gender.

```python
grouped = data.groupby("Gender")["Score"].mean()
print(grouped)
```

**Goal:** Learn grouping and aggregation, very useful in data summaries.

---

## 🔗 Related Files

* [Day11\_PandasBasics/Day11\_PandasbBasics.py](../Day11_PandasBasics/Day11_PandasBasics.py) - All examples in one script
* [students.csv](../Day11_Pandas_Basics/students.csv) - Sample data file

---

## 🧠 Summary

| Concept            | Description                  |
| ------------------ | ---------------------------- |
| Series             | 1D labeled array             |
| DataFrame          | 2D labeled table of data     |
| read\_csv()        | Load CSV data                |
| iloc\[], \["col"]  | Row and column selection     |
| dropna(), fillna() | Handle missing values        |
| drop\_duplicates() | Remove repeated rows         |
| groupby()          | Aggregate rows by categories |
| to\_csv()          | Save cleaned data            |

📅 Up next: **Day 12 - Data Cleaning and Feature Engineering** - Learn how to prepare real-world data for machine learning!
