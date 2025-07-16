# 📅 Day 15 - Introduction to Machine Learning with Scikit-learn

Welcome to **Day 15** of your AI-90Days journey! Today marks an exciting shift - you're about to build your very first **Machine Learning (ML)** model using the powerful Python library **Scikit-learn**.

Machine Learning allows computers to learn from data and make predictions, without being explicitly programmed. It's used everywhere - from spam filters to Netflix recommendations to self-driving cars.

In this lesson, we'll focus on the **basics of supervised learning**, particularly **Linear Regression**, to predict numeric values like student scores from study hours.

---

## 🌟 Objectives

- Understand the basics of machine learning and Scikit-learn
- Learn the steps of building an ML pipeline
- Build your first model: Linear Regression
- Evaluate model performance using metrics
- Visualize the results
- Practice with hands-on exercises

---

## 🔍 What is Machine Learning?

**Machine Learning** is the practice of using algorithms that can learn from data to make decisions or predictions. Instead of writing rules like `if...else`, we show the machine examples - and it learns patterns by itself.

### 🤖 Real-World Examples

- **Spam Detection**: Classify emails as spam or not spam
- **Price Prediction**: Estimate house prices based on features
- **Recommendation Systems**: Suggest movies or products
- **Medical Diagnosis**: Detect disease based on patient data

### 🧠 Types of Machine Learning

There are three major types of machine learning:

| Type          | Description                                                |
| ------------- | ---------------------------------------------------------- |
| Supervised    | Learn from labeled data (e.g. predict a number or a label) |
| Unsupervised  | Find hidden patterns in data (e.g. clustering, grouping)   |
| Reinforcement | Learn by trial and error (e.g. games, robotics)            |

Today, we'll focus on **Supervised Learning**, the most beginner-friendly and widely used approach.

---

## 🧰 What is Scikit-learn?

**Scikit-learn** is a popular Python library for building and experimenting with ML models. It offers simple APIs for:

- Data splitting
- Preprocessing
- Ready-to-use ML models (like regression, classification, clustering)
- Tools to split data, evaluate performance, and tune models
- Easy integration with Pandas and NumPy

### 🔧 Setup

Make sure you have the required libraries installed:

```python
pip install scikit-learn pandas
```

## 📈 Your First ML Model: Linear Regression

Let's start with **Linear Regression**, a simple algorithm used to **predict numeric values** (e.g., predicting a student's score based on study hours).

---

## 🧪 Step-by-Step: Linear Regression with Scikit-learn

### 1️⃣ Import Libraries and Prepare Data

Before building a model, we need to load libraries, and define our input (`X`) and output (`y`).

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset: Hours studied vs Score achieved
data = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6],
    "Score": [35, 45, 50, 60, 68, 75]
})

X = data[["Hours"]]   # Input feature
y = data["Score"]     # Target/output
```

#### 🧠 Explanation:

- `X (features)`: the input - here, it's `"Hours studied"`.
- `y (target)`: the output - the `"Score"` we want to predict.
- The `X` must be 2D (hence `[[ ]]`), and `y` can be 1D.

---

### 2️⃣ Split Data for Training and Testing

We don't want our model to just memorize existing data, we want it to learn patterns and generalize to unseen data.

To build a reliable model, we split our data into two parts:

- **Training set** - to teach the model
- **Test set** - to evaluate how well it learned

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### 🧠 Explanation:

- `train_test_split()` randomly divides data into training and test sets.
- `test_size=0.2` means 20% of data is for testing.
- `random_state` ensures reproducible results.

---

### 3️⃣ Train the Model

Training means letting the model learn from our data by finding the best fit line between inputs and outputs. Now that we have data, we use **Linear Regression** to train the model.

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

#### 🧠 Explanation:

- `LinearRegression()` creates a model that tries to draw the best-fitting straight line.
- `.fit()` teaches the model using the training data (`X` and `y`) so it can learn that relationship.

---

### 4️⃣ Make Predictions

Once trained, the model can now predict outputs (scores) from new inputs (hours). We can use the model to make predictions, just like asking:
*"What score will someone get if they study 3 hours?"*

```python
predictions = model.predict(X_test)
print("Predicted Scores:", predictions)
```

#### 🧠 Explanation:

- `.predict()` generates predictions for unseen data (`X_test`).
- These predictions can then be compared to real `y_test` values to see how accurate it is.

---

### 5️⃣ Evaluate the Model

We measure how good our predictions are using **evaluation metrics**.

```python
from sklearn.metrics import mean_squared_error, r2_score

print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R² Score:", r2_score(y_test, predictions))
```

#### 🧠 Explanation:

- **Mean Squared Error (MSE)**: Measures average squared difference between predicted and actual values (lower = better).
- **R² Score**: Ranges from 0 to 1. A higher value means the model fits well.

---

### 💡 Bonus: Predict for New Students

What if a student studied 1.5 or 7 hours? Let’s try predicting on fresh values. You can create a new **DataFrame** with the same structure (i.e. one column `"Hours"`), just like the original input:

```python
# New data (same structure as training data)
new_data = pd.DataFrame({
    "Hours": [1.5, 3.5, 5.5, 7]   # You can test any values
})

# Predict the scores
predicted_scores = model.predict(new_data)

# Print results
print("\nPredicted Scores:", predicted_scores)
```

#### 🧠 Explanation

You can pass any new hours (same column name) and the model will predict scores. This simulates real-world usage.

- `new_data` must be a **DataFrame** with column **name** `"Hours"`, same as training.
- `model.predict()` uses the learned relationship between **Hours** and **Score** to give predictions.
- This simulates giving the model new students and asking: *"What would be their score if they study X hours?"*

---

## 🧪 Practice Exercises

These exercises help reinforce the full pipeline from building a model to evaluating it and making predictions.

### ✅ Practice Exercise 1: Build & Evaluate a Regression Model

Try working with a slightly larger dataset and use a different evaluation metric (MAE). Let's build a model with a larger dataset to better understand each step.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Create dataset
df = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Score": [32, 45, 50, 60, 65, 72, 80, 90]
})

# Step 2: Define input and output
X = df[["Hours"]]
y = df["Score"]

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Step 4: Train model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = reg.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
```

#### 🧠 Explanation:

- We're following all ML steps using a real dataset.
- **MAE** (Mean Absolute Error) is another metric, measuring average error without squaring. It is the average of absolute errors.
- It’s easier to interpret compared to MSE.

---

### ✅ Practice Exercise 2: Visualize the Regression Line

Visuals help you better understand how well the model fits.

```python
import matplotlib.pyplot as plt

# Plot original data
plt.scatter(X, y, color="blue", label="Actual Data")

# Plot the regression line
plt.plot(X, reg.predict(X), color="red", label="Regression Line")

plt.title("Study Hours vs Score")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.legend()
plt.show()
```

#### 🧠 Explanation:

- Blue dots = actual data
- Red line = predicted line
- If red line passes close to blue dots, model is doing well.

---

### ✅ Practice Exercise 3: Make Your Own Prediction

Try predicting the score if someone studies 9.5 hours:

```python
predicted = reg.predict([[9.5]])
print("Predicted Score for 9.5 hours:", predicted)
```

#### 🧠 Explanation:

- Input must be a 2D array: `[[9.5]]`.
- This simulates a real scenario: *"If I study 9.5 hours, what's my score?"*


---

## 🔗 Related Files

- [Day15_Machine_Learning/Day15_Machine_Learning.py](../Day15_Machine_Learning/Day15_Machine_Learning/Day15_Machine_Learning.py) - All code examples

---


## 🧠 Summary

| Step              | Description                              |
| ----------------- | ---------------------------------------- |
| Import libraries  | Use Scikit-learn, Pandas, Matplotlib     |
| Prepare data      | Organize features (`X`) and labels (`y`) |
| Split dataset     | Use `train_test_split()`                 |
| Train model       | Use `LinearRegression().fit()`           |
| Predict           | Use `.predict()`                         |
| Evaluate model    | Use `MSE`, `MAE`, `R²` metrics           |
| Visualize results | Plot regression line using `matplotlib`  |


---

## 🎁 Bonus: Build Your First ML Model

Let's build our first ML model using complex data. We'll use the Iris Dataset to classify flowers based on their features.

---

### 🌼 Step 1: Load the Dataset

We'll start with the **Iris dataset**, a built-in dataset used for classifying flowers into 3 species based on their petal and sepal dimensions.

```python
from sklearn.datasets import load_iris
import pandas as pd

# Load the dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target

# View first few rows
print(df.head())
```

#### 💡 Explanation:

- `load_iris()` loads a small sample dataset with features like sepal/petal length and width.
- `target` represents the **species** (0 = setosa, 1 = versicolor, 2 = virginica).
- We convert the data into a pandas DataFrame for easier handling.

---

### ✂️ Step 2: Split the Data

Before training, we need to split the dataset into a **training set** (to learn from) and a **test set** (to evaluate performance).

```python
from sklearn.model_selection import train_test_split

X = df[iris.feature_names]  # Features
y = df["target"]            # Labels (species)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### 💡 Explanation:

- `X` contains the features (the input data).
- `y` contains the labels (species we want to predict).
- `train_test_split()` splits the dataset: 80% training and 20% testing.

---

### 🤖 Step 3: Train a Machine Learning Model

Now, we'll train a **K-Nearest Neighbors (KNN)** model using the training data.

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
```

#### 💡 Explanation:

- `KNeighborsClassifier(n_neighbors=3)` means the model looks at the 3 nearest points when predicting.
- `.fit()` trains the model on the training data (`X_train` and `y_train`).

---

### 🔍 Step 4: Make Predictions

Now that our model is trained, let's use it to make predictions on unseen data (the test set).

```python
predictions = model.predict(X_test)
print("Predicted labels:", predictions)
```

#### 💡 Explanation:

- `.predict()` uses the trained model to guess the species of flowers in the test set.
- These are the predicted class labels (0, 1, or 2).

---

### 📈 Step 5: Evaluate the Model

We'll now calculate how accurate the model is by comparing predictions to the actual labels.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
```

#### 💡 Explanation:

- `accuracy_score()` compares predicted vs. actual labels.
- The result is the **percentage of correct predictions** (e.g., 0.96 = 96%).

---

## 📘 Summary of the ML Pipeline

| Step        | Action                       |
| ----------- | ---------------------------- |
| Load data   | `load_iris()`                |
| Prepare X/y | Separate features and labels |
| Split data  | `train_test_split()`         |
| Train model | `model.fit()`                |
| Predict     | `model.predict()`            |
| Evaluate    | `accuracy_score()`           |

## 🧪 Practice Exercises
### ✅ Exercise 1: Try a Different Classifier – Decision Tree

Let's train a Decision Tree instead of KNN and compare results.
```python
from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

tree_preds = tree_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, tree_preds))
```

#### 💡 Explanation:

- Decision Trees split the data into decision paths.
- They are easy to interpret and visualize.
- Accuracy may vary depending on the model and data.

---

### ✅ Exercise 2: Try Another Dataset – Breast Cancer Detection

Use the Breast Cancer dataset, another built-in dataset.

```python
from sklearn.datasets import load_breast_cancer

# Load data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# Split
X = df[data.feature_names]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
preds = model.predict(X_test)
print("Cancer Dataset Accuracy:", accuracy_score(y_test, preds))
```

#### 💡 Explanation:

- The target labels are: 0 = malignant, 1 = benign.
- You follow the same ML pipeline on a new dataset.
- This demonstrates how **reusable and consistent** the process is.

---

### ✅ Exercise 3: Try Different k Values in KNN

Let's experiment with different values of k to see how it impacts accuracy.

```python
for k in range(1, 6):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"K={k} ➤ Accuracy: {acc:.2f}")
```

#### 💡 Explanation:

- A lower `k` value can lead to **overfitting** (too sensitive).
- A higher `k` value smooths out noise but may **underfit**.
- It's good practice to test multiple configurations.

## 🔗 Related Files

- [Day15_Machine_Learning/Day15_Machine_Learning2.py](../Day15_Machine_Learning/Day15_Machine_Learning/Day15_Machine_Learning2.py) - All code examples

---

📘 Summary

| Concept                | Description                                             |
| ---------------------- | ------------------------------------------------------- |
| Scikit-learn           | Python library for ML with tools for the full pipeline  |
| load\_iris(), load\_\* | Preloaded datasets for learning and testing             |
| KNN Classifier         | Simple model that predicts by comparing nearby examples |
| Decision Tree          | A tree-based model that makes decisions at each branch  |
| train\_test\_split()   | Divides dataset into training and testing sets          |
| accuracy\_score()      | Measures the percentage of correct predictions          |
| Breast Cancer Dataset  | Real-world binary classification task                   |


📅 Up next: **Day 16 - Data Preprocessing and Pipelines in Scikit-learn**, where we move from predicting numbers to predicting categories like Pass/Fail or Yes/No.