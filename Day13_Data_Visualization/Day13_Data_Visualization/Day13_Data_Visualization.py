import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Line Plot
x = [1, 2, 3, 4, 5]
y = [10, 15, 13, 17, 20]
plt.plot(x, y)
plt.title("Line Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()

# Bar Plot
categories = ["Math", "Science", "English"]
scores = [85, 90, 78]
plt.bar(categories, scores, color="skyblue")
plt.title("Student Scores")
plt.xlabel("Subjects")
plt.ylabel("Marks")
plt.show()

# Histogram
data = np.random.normal(60, 10, 100)
plt.hist(data, bins=10, color='green')
plt.title("Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# Histogram with KDE
data = [55, 60, 65, 70, 75, 80, 85, 90, 95]

sns.histplot(data, kde=True)
plt.title("Histogram with KDE")
plt.show()

# Barplot
data = pd.DataFrame({
    "Subject": ["Math", "Science", "English", "History"],
    "Score": [88, 92, 81, 77]
})

sns.barplot(x="Subject", y="Score", data=data)
plt.title("Student Scores by Subject")
plt.show()

# Line Plot
# Monthly temperature data
months = ["Jan", "Feb", "Mar", "Apr", "May"]
temps = [15, 18, 21, 25, 30]

sns.lineplot(x=months, y=temps)
plt.title("Monthly Average Temperatures")
plt.show()

# Scatter Plot
df = pd.DataFrame({
    "Hours_Studied": [1, 2, 3, 4, 5],
    "Exam_Score": [50, 55, 65, 70, 85]
})
sns.scatterplot(data=df, x="Hours_Studied", y="Exam_Score")
plt.title("Study Time vs Score")
plt.show()

# Box Plot
scores = [40, 45, 50, 60, 65, 70, 75, 90, 100]
sns.boxplot(y=scores)
plt.title("Box Plot of Scores")
plt.show()

# Heatmap
df = pd.DataFrame({
    "Math": [80, 90, 70],
    "Science": [85, 88, 78],
    "English": [75, 70, 72]
})
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Subject Correlation Heatmap")
plt.show()

# Heatmap without annotations
# Example 2D data
data = np.array([[90, 80, 70],
                 [65, 75, 85],
                 [50, 60, 95]])

sns.heatmap(data)
plt.title("Student Scores Heatmap")
plt.show()

# Heatmap with annotations
# Example 2D data
data = np.array([[90, 80, 70],
                 [65, 75, 85],
                 [50, 60, 95]])

sns.heatmap(data, annot=True)
plt.title("Student Scores Heatmap")
plt.show()

# Load the data
df = pd.read_csv("students.csv")
print("Original Data:")
print(df)

# Practice 1: Compare Male vs Female Scores (Bar Plot)
data = pd.DataFrame({
    "Gender": ["Male", "Female", "Male", "Female", "Male"],
    "Score": [70, 85, 60, 90, 75]
})

avg_scores = data.groupby("Gender")["Score"].mean().reset_index()
sns.barplot(data=avg_scores, x="Gender", y="Score")
plt.title("Average Scores by Gender")
plt.show()

#Practice 2: Visualize Score Distribution (Histogram)
scores = [45, 50, 55, 60, 60, 65, 70, 75, 80, 90]

plt.hist(scores, bins=5, color="orange")
plt.title("Histogram of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# Practice Exercise 3: Correlation Heatmap of Student Subjects
df_subject = pd.DataFrame({
    "Math": [85, 70, 90, 65, 95],
    "Science": [80, 75, 88, 60, 92],
    "English": [78, 72, 84, 68, 88],
    "History": [82, 76, 85, 65, 90]
})

sns.heatmap(df_subject.corr(), annot=True, cmap="YlGnBu")
plt.title("Student Subjects Correlation")
plt.show()

# Practice 4: Gender-wise Average Score Bar Plot
avg_scores = df.groupby("Gender")["Score"].mean()
print("\nAverage Score by Gender:")
print(avg_scores)

avg_scores.plot(kind="bar", title="Average Score by Gender", color=["blue", "purple"])
plt.ylabel("Average Score")
plt.xticks(rotation=0)
plt.show()

# Practice 5: Score Distribution Histogram
plt.hist(df["Score"], bins=8, color="orange", edgecolor="black")
plt.title("Score Distribution")
plt.xlabel("Score")
plt.ylabel("Number of Students")
plt.grid(axis="y")
plt.show()

# Practice 6: Correlation Heatmap
print("\nCorrelation Matrix:")
numeric_df = df.select_dtypes(include=[np.number])
print(numeric_df.corr())
sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Practice 7: Box Plot by Gender
sns.boxplot(data=df, x="Gender", y="Score", palette="pastel")
plt.title("Score Distribution by Gender")
plt.show()

# Practice 8: Scatter Plot – Age vs Score
sns.scatterplot(data=df, x="Age", y="Score", hue="Gender", style="Gender")
plt.title("Age vs Score by Gender")
plt.xlabel("Age")
plt.ylabel("Score")
plt.show()

# Practice 9: Count Plot for City
sns.countplot(data=df, x="City", palette="Set2")
plt.title("Number of Students by City")
plt.xticks(rotation=30)
plt.show()

# Practice 10: Create a Pie Chart of Favorite Fruits

labels = ["Apple", "Banana", "Cherry", "Orange"]
sizes = [30, 25, 20, 25]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Favorite Fruits Distribution")
plt.show()

# Practice 11: Visualize Gender Distribution

gender_data = pd.DataFrame({"Gender": ["Male"]*60 + ["Female"]*40})
sns.countplot(x="Gender", data=gender_data)
plt.title("Gender Distribution")
plt.show()

print("Data Visualization Completed Successfully!")