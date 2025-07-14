# Day 13 - Data Visualization with Matplotlib & Seaborn 📈🎨

Welcome to **Day 13** of your AI-90Days journey! Today, you'll learn how to **visualize your data** using two powerful Python libraries: **Matplotlib** and **Seaborn**. Visualizing data is one of the best ways to explore patterns, detect anomalies, and communicate insights clearly.

---

## 🌟 Objectives

- Learn basic plots with Matplotlib and Seaborn
- Explore patterns and trends in the dataset visually
- Use bar charts, histograms, scatter plots, heatmaps, and more
- Customize plots (colors, labels, styles)
- Practice with real-world student data

---

## 🔍 Why Data Visualization?

Data visualization helps transform raw data into visuals that can be interpreted quickly. It's easier to detect patterns, relationships, and outliers using charts than by reading numbers.

 - **Matplotlib** and **Seaborn** are Python libraries used for data visualization, helping you turn raw data into charts, plots, and graphs.
 - Together, they help you **see trends, patterns, and insights in your data**, which is crucial for any AI, ML, or data science project.

---

## 📦 Setup

Make sure you have the required libraries installed:

```bash
pip install matplotlib seaborn pandas
```

Create a new project called `Day13_Data_Visualization` and use the `students.csv` file provided.

---

## Introduction to Matplotlib

**Matplotlib** is the most widely used basic plotting library in Python. It lets you create bar charts, line plots, histograms, pie charts, and much more. It gives you full control over your plot’s appearance.

---

### 📈 Line Plot

Line plots are useful for showing changes over time or a sequence.

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 15, 13, 17, 20]

plt.plot(x, y)
plt.title("Line Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()
```

**Explanation:**

* `plot(x, y):` Draws a line connecting the points
* `title`, `xlabel`, `ylabel`: Add labels and title
* `grid(True):` Adds a grid for better readability

---

### 📃 Bar Plot

Bar plots help you compare values across categories.

```python
categories = ["Math", "Science", "English"]
scores = [85, 90, 78]

plt.bar(categories, scores, color="skyblue")
plt.title("Student Scores")
plt.xlabel("Subjects")
plt.ylabel("Marks")
plt.show()
```

**Explanation:**

* `bar()` is used for vertical bars
* Great for comparing categories like scores or counts

---

### 📉 Histogram

Histograms show how your data is distributed.

```python
import numpy as np

data = np.random.normal(60, 10, 100)

plt.hist(data, bins=10, color='green')
plt.title("Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()
```

**Explanation:**

* `hist()`: Breaks the data into intervals (bins)
* Shows how frequently values fall into those ranges

---

### 🧠 What is KDE?

KDE stands for **Kernel Density Estimate**. It's a smoothed line that estimates the probability distribution of a continuous variable - basically a smoothed version of your histogram.

```python
import seaborn as sns
import matplotlib.pyplot as plt

data = [55, 60, 65, 70, 75, 80, 85, 90, 95]

sns.histplot(data, kde=True)
plt.title("Histogram with KDE")
plt.show()
```

---

### 🔍 What does `kde=True` do?

* Draws the histogram bars to show how often values occur.
* Adds a smooth line curve (KDE) that tries to represent the overall distribution pattern of the data.

---

### 📌 Why use KDE?

* It helps you see the shape of the data distribution more clearly than a histogram alone.
* Especially useful when you want to know if data is skewed, bimodal, or roughly normal.

---

## 🌟 Introduction to Seaborn

**Seaborn** is built on top of **Matplotlib** and offers simpler syntax and beautiful styles by default.

 - It makes beautiful plots easily and adds more powerful features like color themes, advanced statistical plots, and better default styles. 
 - It's perfect for analyzing relationships between variables (like scatter plots, box plots, and heatmaps).

### Bar Plot

Bar plot displays the result visually so we can easily compare the data.

```python
import seaborn as sns
import pandas as pd

# Sample Data
data = pd.DataFrame({
    "Subject": ["Math", "Science", "English", "History"],
    "Score": [88, 92, 81, 77]
})

sns.barplot(x="Subject", y="Score", data=data)
plt.title("Student Scores by Subject")
plt.show()
```

**Explanation:**

* `sns.barplot()` draws a bar chart with better aesthetics than Matplotlib
* Works directly with Pandas DataFrames

---

### 🔢 Line Plot

Line plot shows trends over time 

```python
# Monthly temperature data
months = ["Jan", "Feb", "Mar", "Apr", "May"]
temps = [15, 18, 21, 25, 30]

sns.lineplot(x=months, y=temps)
plt.title("Monthly Average Temperatures")
plt.show()
```

**Explanation:**

* `sns.lineplot()` creates a line chart with automatic smoothing and styling

---

## 📌 Scatter Plot 

Scatter plots help you see the relationship between two numeric variables.


```python
import seaborn as sns
import pandas as pd

df = pd.DataFrame({
    "Hours_Studied": [1, 2, 3, 4, 5],
    "Exam_Score": [50, 55, 65, 70, 85]
})

sns.scatterplot(data=df, x="Hours_Studied", y="Exam_Score")
plt.title("Study Time vs Score")
plt.show()
```

**Explanation:**

* `scatterplot()`: Creates a dot for each pair
* Useful to see if more study leads to better scores (positive trend)

---

## 📦 Box Plot 

Box plots show the spread and outliers of data.


```python
scores = [40, 45, 50, 60, 65, 70, 75, 90, 100]

sns.boxplot(y=scores)
plt.title("Box Plot of Scores")
plt.show()
```

**Explanation:**

* Box plot displays the `median`, `quartiles`, and `outliers`
* Helps detect extreme values

---

## 🌡️ Heatmap 

Heatmaps help you visualize correlations between variables.

```python
df = pd.DataFrame({
    "Math": [80, 90, 70],
    "Science": [85, 88, 78],
    "English": [75, 70, 72]
})

correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Subject Correlation Heatmap")
plt.show()
```

**Explanation:**

* `corr()` calculates correlation between columns
* `heatmap()` colors the matrix to show strength of relationships
* `annot` stands for **annotation**. When you set `annot=True`, Seaborn will write the actual data values inside the boxes of the heatmap.

---

#### 🔍 Without `annot=True`

You will see only color-coded squares (like a colored grid), but no numbers inside them.

```python
# Example 2D data
data = np.array([[90, 80, 70],
                 [65, 75, 85],
                 [50, 60, 95]])

sns.heatmap(data)
plt.title("Student Scores Heatmap")
plt.show()
```

---

#### 🔍 With `annot=True`

You get the same colored grid plus the actual numbers displayed inside each square.

```python
# Example 2D data
data = np.array([[90, 80, 70],
                 [65, 75, 85],
                 [50, 60, 95]])

sns.heatmap(data, annot=True)
plt.title("Student Scores Heatmap")
plt.show()
```

---

#### 📌 Why use annot=True?

* Makes the heatmap more informative and readable.
* Allows you (and others) to **see exact values** while still getting a visual sense of data patterns via color.

---

## ✅ Practice Exercise 1: Compare Male vs Female Scores (Bar Plot)

In this exercise, we will calculate the average score of male and female students, then visualize the comparison using a bar plot. This helps us compare performance between categories.

```python
data = pd.DataFrame({
    "Gender": ["Male", "Female", "Male", "Female", "Male"],
    "Score": [70, 85, 60, 90, 75]
})

avg_scores = data.groupby("Gender")["Score"].mean().reset_index()
sns.barplot(data=avg_scores, x="Gender", y="Score")
plt.title("Average Scores by Gender")
plt.show()
```

** Explanation:** 

* `groupby("Gender")["Score"].mean()` calculates the average score for each gender.
* `reset_index()` converts the result back to a DataFrame for plotting.
* `sns.barplot()` displays the result visually so we can easily compare male vs. female averages.

---

## ✅ Practice Exercise 2: Visualize Score Distribution (Histogram)

Histograms help us understand how values (like scores) are distributed across a dataset. See how scores are spread out.

```python
scores = [45, 50, 55, 60, 60, 65, 70, 75, 80, 90]

plt.hist(scores, bins=5, color="orange")
plt.title("Histogram of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()
```

**Explanation:**

* This groups scores into ranges (bins) and counts how many fall in each range.
* Great for spotting patterns like skewness, peaks, or gaps in the data.

---

## ✅ Practice Exercise 3: Correlation Heatmap of Student Subjects

Here, we examine how subject scores relate to one another. Do students who score well in Math also score well in Science?

Let's build a more detailed subject dataset and analyze the correlation.

```python
df_subject = pd.DataFrame({
    "Math": [85, 70, 90, 65, 95],
    "Science": [80, 75, 88, 60, 92],
    "English": [78, 72, 84, 68, 88],
    "History": [82, 76, 85, 65, 90]
})

sns.heatmap(df_subject.corr(), annot=True, cmap="YlGnBu")
plt.title("Student Subjects Correlation")
plt.show()
```

**Explanation:**

* Helps detect if high performance in one subject relates to another
* `.corr()` calculates correlation between all numeric columns.
* A heatmap makes it easy to spot strong or weak relationships between subjects.
* `annot=True` displays the actual numbers inside each cell.
---

## ✅ Practice Exercise 4: Gender-wise Average Score Bar Plot

Now we'll analyze real data from a CSV to compare how male and female students perform on average.

```python
df = pd.read_csv("students.csv")
avg_scores = df.groupby("Gender")["Score"].mean()

avg_scores.plot(kind="bar", title="Average Score by Gender", color=["blue", "purple"])
plt.ylabel("Average Score")
plt.xticks(rotation=0)
plt.show()
```

**Explanation:**  

* We calculate average score per gender and visualize it using a bar plot.
* Easy way to compare student performance by gender using real-world data.

---

## ✅ Practice Exercise 5: Score Distribution with Histogram

This histogram gives us a real-world view of how students are performing overall. Understand how scores are distributed using a histogram.

```python
plt.hist(df["Score"], bins=8, color="orange", edgecolor="black")
plt.title("Score Distribution")
plt.xlabel("Score")
plt.ylabel("Number of Students")
plt.grid(axis="y")
plt.show()
```

**Explanation:**  

Histogram groups scores into bins to show frequency.

* Divides scores into 8 bins and counts students in each.
* Edge color improves readability.
* Useful for checking if most students score within a certain range.

---

## ✅ Practice Exercise 6: Correlation Heatmap

This shows how different numeric features (like Age, Score, etc.) relate to each other.

```python
numeric_df = df.select_dtypes(include=[np.number])
print(numeric_df.corr())
sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()
```

**Explanation:**  

* Correlation matrix shows relationships between numerical columns.
* Helps you detect useful relationships (or lack of them) between columns in your dataset.
* High correlation between features may lead to feature redundancy in machine learning.

---

## ✅ Practice Exercise 7: Box Plot by Gender

Box plots reveal score distribution and highlight medians and outliers.

```python
sns.boxplot(data=df, x="Gender", y="Score", palette="pastel")
plt.title("Score Distribution by Gender")
plt.show()
```

**Explanation:**  

* Box plots help visualize data distribution and detect outliers.
* Shows the median (middle line), quartiles, and any outliers.
* Useful for comparing spread and consistency between genders..

---

## ✅ Practice Exercise 8: Scatter Plot - Age vs Score

We visualize the relationship between a student's age and their score, color-coded by gender.

```python
sns.scatterplot(data=df, x="Age", y="Score", hue="Gender", style="Gender")
plt.title("Age vs Score by Gender")
plt.xlabel("Age")
plt.ylabel("Score")
plt.show()
```

**Explanation:**  

* This scatter plot helps identify trends or clusters between age and score.
* Shows how score varies with age and if any patterns exist.
* Color and marker style (hue, style) help separate male/female trends.

---

## ✅ Practice Exercise 9: Count Plot for City

Let's see how many students belong to each city.

```python
sns.countplot(data=df, x="City", palette="Set2")
plt.title("Number of Students by City")
plt.xticks(rotation=30)
plt.show()
```

**Explanation:**  

* Displays number of students from each city.
* Count plots are great for visualizing the frequency of categories.
* Rotate labels for better visibility with long city names.

---

## ✅ Practice Exercise 10. Create a Pie Chart of Favorite Fruits

Pie charts help visualize proportions. Let's say we surveyed students on their favorite fruit.

```python
labels = ["Apple", "Banana", "Cherry", "Orange"]
sizes = [30, 25, 20, 25]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Favorite Fruits Distribution")
plt.show()
```

* Each slice represents a category's proportion in the whole.
* `autopct` shows percentage, and `startangle=90` rotates the chart so the first slice starts at the top.

---

## ✅ Practice Exercise 11. Visualize Gender Distribution

This simple count plot shows how many males and females are in the dataset.

```python
gender_data = pd.DataFrame({"Gender": ["Male"]*60 + ["Female"]*40})
sns.countplot(x="Gender", data=gender_data)
plt.title("Gender Distribution")
plt.show()
```

* Easy visual way to compare category counts.
* Useful for checking data imbalance in classification problems.

---

## 🔗 Related Files

* [Day13_Data_Visualization/Day13_Data_Visualization.py](../Day13_Data_Visualization/Day13_Data_Visualization.py) - All examples in one script

---

## 🧠 Summary

| Plot Type    | Description                                |
| ------------ | ------------------------------------------ |
| Line Plot    | Shows trends over time                     |
| Bar Chart    | Compares categories                        |
| Histogram    | Shows distribution of values               |
| Scatter Plot | Shows relationship between variables       |
| Heatmap      | Displays correlations using colors         |
| Pairplot     | Matrix of relationships and distributions  |
| Boxplot      | Summarizes distribution & detects outliers |

📅 Up next: **Day 14 - Exploratory Data Analysis (EDA)** - Learn how to deeply explore and understand data before modeling.
