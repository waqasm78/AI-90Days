# 📊 Day 14 - Exploratory Data Analysis (EDA)

Welcome to **Day 14** of the AI-90Days journey! Today, we're diving into **Exploratory Data Analysis (EDA)** — a crucial step in any data science or AI project. EDA helps you understand your dataset’s structure, detect patterns, discover anomalies, and form hypotheses through summary statistics and visualizations.

---

## 🌟 Objectives

- Understand what EDA is and why it matters
- Explore data using Pandas and basic visualization
- Identify patterns, distributions, and relationships
- Handle and identify outliers or missing values
- Use visual tools to enhance data understanding

---

## 🔍 What is EDA?

**Exploratory Data Analysis (EDA)** is the process of analyzing datasets to summarize their main characteristics using statistics and visual methods. Before building any machine learning model, it's important to explore the data to make informed decisions about cleaning, transformation, and model selection.

---

## 📁 Load the Dataset

We’ll begin by loading a sample dataset of students.

```python
import pandas as pd

df = pd.read_csv("students.csv")
print(df.head())
```

**Explanation:** We load the dataset using `read_csv()` and view the first few rows with `head()` to understand its structure.

---

## 📐 Step 1: Basic Info and Shape

Start with quick insights into your dataset’s size and structure.

```python
print("Shape:", df.shape)
print(df.info())
```

**Explanation:**  
- `shape`: returns (rows, columns)
- `info()`: gives an overview of column types and non-null counts

---

## 📊 Step 2: Summary Statistics

Check basic statistical details like mean, median, min, max, etc.

```python
print(df.describe())
```

**Explanation:**  
`describe()` provides summary statistics for all numeric columns.

---

## 🔁 Step 3: Checking for Missing Values

Identify any gaps in your dataset.

```python
print(df.isnull().sum())
```

**Explanation:**  
Use `isnull().sum()` to count how many missing entries exist in each column.

---

## 🧍 Step 4: Categorical Value Counts

Understand the distribution of categorical data.

```python
print(df["Gender"].value_counts())
print(df["City"].value_counts())
```

**Explanation:**  
Find out how many males vs females or how many students from each city.

---

## 🧪 Step 5: Correlation Analysis

Explore how numeric columns relate to each other.

```python
print(df.corr(numeric_only=True))
```

**Explanation:**  
`corr()` helps spot trends like whether higher age correlates with higher scores.

---

## ✅ Practice Exercise 1: Gender-wise Summary

Let’s explore score distributions and counts grouped by gender.

```python
summary = df.groupby("Gender")["Score"].describe()
print(summary)
```

**Explanation:**  
This grouped summary gives min, max, mean, std, etc., for male vs female students.

---

## ✅ Practice Exercise 2: Average Score by City

Which city has the highest average student score?

```python
avg_by_city = df.groupby("City")["Score"].mean()
print(avg_by_city)
```

**Explanation:**  
We use `groupby()` to group data by City and compute the mean of the Score column.

---

## ✅ Practice Exercise 3: Find Students with Highest and Lowest Score

```python
top_student = df[df["Score"] == df["Score"].max()]
low_student = df[df["Score"] == df["Score"].min()]

print("Top Performer:
", top_student)
print("Lowest Performer:
", low_student)
```

**Explanation:**  
Use filtering and max()/min() to locate students who scored the highest and lowest.

---

## ✅ Practice Exercise 4: Detect Outliers

Use interquartile range (IQR) to identify score outliers.

```python
Q1 = df["Score"].quantile(0.25)
Q3 = df["Score"].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df["Score"] < (Q1 - 1.5 * IQR)) | (df["Score"] > (Q3 + 1.5 * IQR))]
print("Outliers:
", outliers)
```

**Explanation:**  
The IQR method is used to detect data points that are significantly higher or lower than the rest.

---

## ✅ Practice Exercise 5: Create a New Column for Performance

Bin the students into categories like Low, Medium, High.

```python
bins = [0, 60, 80, 100]
labels = ["Low", "Medium", "High"]
df["Performance"] = pd.cut(df["Score"], bins=bins, labels=labels)

print(df[["Name", "Score", "Performance"]])
```

**Explanation:**  
We classify scores into labeled bins using `pd.cut()`.

---

## 🔗 Related Files

- [Day14_EDA/Day14_EDA.py](../Day14_EDA/Day14_EDA/Day14_EDA.py) - All code examples
- [students.csv](../Day14_EDA/Day14_EDA/students.csv) - Sample dataset

---

## 🧠 Summary

| Step                       | Description                                        |
|----------------------------|----------------------------------------------------|
| `df.head()`                | Preview dataset                                    |
| `df.info()` / `df.describe()` | Overview and summary stats                    |
| `value_counts()`           | Frequency of categories                           |
| `groupby()`                | Segment and aggregate data                        |
| `corr()`                   | Find relationships                                |
| `pd.cut()`                 | Bin numeric values into categories                |
| IQR                        | Outlier detection                                 |

📅 Up next: **Day 15 - Introduction to Machine Learning with Scikit-learn**

