# Day 12 - Data Cleaning and Feature Engineering 📊✨

Welcome to **Day 12** of your AI-90Days journey! Today is all about preparing your data so that it's ready for machine learning models. This is a crucial step in every data science project, often taking up to **80% of your time**. The quality of your data directly impacts your model performance.

---

## 🌟 Objectives

* Understand the importance of clean data
* Learn techniques to clean and preprocess data
* Explore common data cleaning tasks (e.g., filling missing values, encoding categories)
* Learn feature engineering concepts and how to apply them
* Practice hands-on with examples

---

## 🔍 What is Data Cleaning?

**Data cleaning** is the process of fixing or removing incorrect, corrupted, or incomplete data within your dataset. In real-world data, you're likely to deal with typos, missing values, duplicates, inconsistent formats, and outliers. Cleaning ensures your models can learn patterns effectively.

---

## 🌀 Removing Duplicates and Invalid Entries

Removing duplicates is often the first step. Also, you may want to filter out invalid data like negative ages or impossible values.

```python
import pandas as pd

df = pd.read_csv("students.csv")
df = df.drop_duplicates()
df = df[df["Age"] >= 0]  # Remove invalid age rows
```

**Explanation:**

* `drop_duplicates()` removes repeated rows.
* `df[df["Age"] >= 0]` filters out entries where age is negative.

---

## 📃 Handling Missing Values

Missing values are extremely common. There are several strategies to handle them:

```python
# Example with missing scores
print(df.isnull().sum())  # Check how many missing per column

df["Score"] = df["Score"].fillna(df["Score"].mean())
```

**Explanation:**

* `isnull().sum()` helps identify where data is missing.
* `fillna()` replaces missing values with a substitute. In this case, we used the column mean.

---

## 🧹 Standardizing Text Data

Text entries might vary in case, spelling, or extra spaces. You can clean them by converting to lowercase, trimming spaces, or replacing misspelled values.

```python
df["City"] = df["City"].str.lower().str.strip()
df["City"] = df["City"].replace({"lahor": "lahore"})
```

**Explanation:**

* `.str.lower()` converts all entries to lowercase.
* `.str.strip()` removes extra whitespace.
* `.replace()` fixes common misspellings.

---

## 🔢 Encoding Categorical Variables

ML models work with numbers, not strings. So you must convert categories like "Male", "Female" into numeric values.

```python
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
```

**Explanation:**

* `map()` replaces category labels with numbers for model compatibility.

---

## 🤑 Feature Engineering Basics

**Feature engineering** is the process of creating new features from existing ones to better capture patterns in data.

For example, we can create a "Result" column based on the score:

```python
df["Result"] = df["Score"].apply(lambda x: "Pass" if x >= 50 else "Fail")
```

**Explanation:**

* `.apply()` with `lambda` lets you write custom logic to generate new columns.

---

## 🔄 Binning (Bucketing Values)

Sometimes it's useful to group numerical data into ranges or categories (like "Low", "Medium", "High").

```python
bins = [0, 50, 70, 100]
labels = ["Low", "Medium", "High"]
df["Performance"] = pd.cut(df["Score"], bins=bins, labels=labels)
```

**Explanation:**

**1.** `bins = [0, 50, 70, 100]`

This defines the numeric ranges for binning. These are the **edges** of the intervals.

It creates **3 intervals**:

- 0 < score ≤ 50
- 50 < score ≤ 70
- 70 < score ≤ 100

So we have **3 bins**:

- Bin 1: between 0 and 50
- Bin 2: between 50 and 70
- Bin 3: between 70 and 100

    That's why you need 4 numbers to define 3 bins - each pair of consecutive numbers makes 1 bin.

**2.** `labels = ["Low", "Medium", "High"]`

These are the **names** you want to give to each bin:

- `"Low"` -> for scores between 0-50
- `"Medium"` -> for scores between 50-70
- `"High"` -> for scores between 70-100

So, the number of labels must be one less than the number of bin edges.

**3.** `pd.cut(df["Score"], bins=bins, labels=labels)`

- `pd.cut()` assigns each `Score` in the DataFrame to one of the bins.
- It then replaces the score with its corresponding label (`Low`, `Medium`, or `High`) in a new column called `"Performance"`.

---

## ✅ Practice Exercise 1: Identify and Fill Missing Age

Let's find and fix missing age values in the dataset.

```python
missing_age_count = df["Age"].isnull().sum()
print("Missing Age:", missing_age_count)

df["Age"] = df["Age"].fillna(df["Age"].median())
```

**Explanation:**

* Use `.median()` if the data has outliers, since it's more robust than the mean.

---

## ✅ Practice Exercise 2: Create Age Groups

Divide students into age categories to make analysis easier.

```python
bins = [0, 18, 25, 100]
labels = ["Teen", "Young Adult", "Adult"]
df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
print(df[["Name", "Age", "AgeGroup"]])
```

**Explanation:**

* Helps categorize age into logical groupings for visualizations or modeling.

---

## ✅ Practice Exercise 3: Normalize Scores

To bring all data into a common scale (0 to 1), use normalization:

```python
df["Score_Norm"] = (df["Score"] - df["Score"].min()) / (df["Score"].max() - df["Score"].min())
```

**Explanation:**

* Normalization ensures each feature contributes equally to the model training.

---

## 🔗 Related Files

* [Day12\_Data\_Cleaning/day12\_data\_cleaning.py](../Day12_Data_Cleaning/day12_data_cleaning.py) - All examples in one script
* [students.csv](../Day11_Pandas_Basics/students.csv) - Sample data file

---

## 🧠 Summary

| Task                     | Method Used                    |
| ------------------------ | ------------------------------ |
| Remove duplicates        | `drop_duplicates()`            |
| Handle missing values    | `fillna()`, `isnull()`         |
| Text cleaning            | `.str.lower()`, `.str.strip()` |
| Encode categories        | `map()`                        |
| Create new features      | `apply()`, `cut()`             |
| Normalize numeric values | `(x - min) / (max - min)`      |

📅 Up next: **Day 13 - Data Visualization with Matplotlib & Seaborn** - Discover how to explore data visually!
