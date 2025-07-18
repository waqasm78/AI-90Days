# 📈 Day 17 - Linear Regression

Welcome to **Day 17** of your AI-90Days journey! Today marks an exciting transition where we dive into our **first full machine learning algorithm: Linear Regression**.

You'll learn what linear regression is, how it works, how to implement it in Scikit-learn, and build a **mini-project** to predict house prices.


---

## 🌟 Objectives

- Understand what Linear Regression is and how it works
- Learn key evaluation metrics: **Mean Squared Error (MSE)** and **R-squared (R²)**
- Train a regression model using `scikit-learn`
- Visualize predictions and evaluate model performance
- Build a mini-project: Predict house prices using linear regression

---

## 🔍 What is Linear Regression?

**Linear Regression** is one of the most fundamental and widely used algorithms in **supervised learning**. It’s often the first step in learning how to model data, and it serves as a building block for understanding more advanced techniques later.

Linear regression tries to establish a **relationship** between **a dependent variable (target)** and one or more **independent variables (features)** by fitting a straight line (in 2D) or a hyperplane (in higher dimensions) to the data.

The goal is simple: "**Given some input features, can we accurately predict the output?**"

---

### 🧠 Intuition Behind Linear Regression

Imagine you're a real estate agent trying to predict the price of a house. You notice that, generally, the more square footage a house has, the higher the price. This is a **linear relationship**.

You can draw a straight line through the data points on a graph:

- X-axis = Square footage
- Y-axis = Price

This line helps predict the price of any house based on its size.

---

### 📏 Formula for Simple Linear Regression:

In the case of a single feature (Simple Linear Regression), the model tries to fit a line like this:

```python
Y = aX + b
```

| Symbol | Meaning                                                                                    |
| ------ | ------------------------------------------------------------------------------------------ |
| `Y`    | The **predicted value** (target)                                                           |
| `X`    | The **feature/input** value                                                                |
| `a`    | The **slope** of the line (coefficient that shows how much `Y` changes when `X` increases) |
| `b`    | The **intercept**, or the value of `Y` when `X = 0`                                        |

This is called **Simple Linear Regression**, because it uses just **one feature (X)**.

---

### 🏠 Example – Predicting House Prices Based on Size

Let's say you have the following dataset showing how the price of houses increases with their size:

| House Size (sqft) | House Price (\$) |
| ----------------- | ---------------- |
| 1000              | 150,000          |
| 1200              | 180,000          |
| 1500              | 225,000          |
| 1800              | 270,000          |

If we fit a **simple linear regression** line to this data, we might get:

```python
Price = 150 * Size + 0
```

This means for every 1 square foot increase, the price goes up by $150. So if a house is 2,000 sqft:

```python
Price = 150 * 2000 = $300,000
```

The model has learned a simple linear pattern and can now make predictions on unseen data.

---

### 📊 What If You Have Multiple Features?

When you use **more than one feature** to predict the target value, it's called **Multiple Linear Regression**.

For example, house price might depend on:

- Size (in square feet)
- Number of bedrooms
- Distance to city center
- Age of the house

The formula becomes:

```python
Price = a1*Size + a2*Bedrooms + a3*Distance + a4*Age + b
```

Each coefficient (a1, a2, etc.) shows how much the price is influenced by that specific feature.

---

### ✅ When to Use Linear Regression?

Use Linear Regression when:

- The relationship between the input features and output is approximately **linear**.
- You want a **simple, fast, and interpretable** model.
- You're working with **numeric features** and want to predict a **continuous value**.

---

### ❌ When Not to Use It?

Avoid Linear Regression when:

- The data shows a **non-linear relationship**.
- You have **outliers** or **correlated features**, which can distort the results.
- You need a model with **very high accuracy** on complex datasets, more advanced models like Random Forest or XGBoost may work better.

---

### 🔧 Setup

Make sure you have the required libraries installed:

```python
pip install scikit-learn pandas
```

---

## 🎁 Build ML Model

Let's build the ML model using a simple simulated dataset that mimics real estate features like **Size of the house** and its corresponding **Price**.

---

### 🏘️ Step 1: Load Dataset

We'll start with a sample dataset that contains `House_Size` and `Price`.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = pd.DataFrame({
    "House_Size": [750, 800, 850, 900, 950, 1000, 1050, 1100],
    "Price": [150000, 160000, 165000, 175000, 180000, 190000, 200000, 210000]
})

print(data)
```

---

### 🧪 Step 2: Split the Data

Before training our model, we need to separate the **features (input)** and **target (output)**, and then split the dataset into a **training set** and a **test set**. This ensures we can train the model on one part and evaluate it on unseen data.

```python
# Separate features (X) and target (y)
X = data[["House_Size"]]  # Feature column (must be 2D)
y = data["Price"]         # Target column

# Split the data into training (75%) and testing (25%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
```

**Explanation:**

- `test_size=0.25` keeps 25% of data for testing and 75% for training.
- `random_state=42` makes the split reproducible.
- This helps us test the model's performance on data it hasn't seen before just like a real-world scenario.

---

#### 📌 Why Do We Split the Data?

Imagine you're studying for a test. If you only ever practice on the exact same questions, you might score high - but only because you memorized the answers. The real test of your understanding is how well you do on new questions you haven't seen before.

That's exactly what we're doing here:

- The **training set** helps the model "study".
- The **test set** checks whether the model truly "understood" or just memorized.

---

### 🧠 Step 3: Train the Linear Regression Model

Once we have split our dataset into training and testing sets and prepared our features (`X_train`) and target (`y_train`), it's time to **train** the linear regression model.

This is where the **learning happens**, the model studies the data and finds the best-fitting line.

```python
# Create the model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)
```

**Explanation:**

- `LinearRegression()` creates an instance of the Linear Regression algorithm
- The `fit()` method tells the model: "Here is the input data (`X_train`) and the correct answers (`y_train`). Learn the best line that fits this data."

Under the hood, the model is learning the values for the following:

- **Slope (`a`)**: This tells the model how much the target value (e.g., house price) changes for each unit change in the feature (e.g., size).
- **Intercept (`b`)**: This tells the model what the predicted target value is when all features are zero.

It finds the **best-fit line** by minimizing the difference between the predicted values and the actual values. This difference is measured using a technique called **Least Squares**, which calculates the line that has the **smallest possible total error** across all training points.

---

### 📈 Step 4: View Learned Parameters

After training the model, we can now check the **slope** and **intercept** it has learned.

```python
print("Slope (Coefficient):", model.coef_)
print("Intercept:", model.intercept_)
```

**Explanation:**

- The **slope (coefficient)** shows how much the price increases for every extra square foot.
- The **intercept** is the predicted base price when the house size is zero.
- Together, they define the best-fit line: `Price = Slope * House_Size + Intercept`

---

### 📊 Step 5: Make Predictions

Now that the model is trained, let's use it to predict house prices based on the test data.

```python
y_pred = model.predict(X_test)
print("Predicted Prices:", y_pred)
```

**Explanation:**

- The model uses the relationship it learned to estimate house prices for unseen data (`X_test`).
- These predictions can now be compared with actual prices to see how well the model performs.

---

### 📏 Step 6: Evaluate the Model

To measure how accurate our predictions are, we use two key metrics: **Mean Squared Error (MSE)** and **R-squared (R^2)**.

```python
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared (R^2):", r2)
```

**Explanation:**

- **MSE**: Average of the squared errors between actual and predicted prices. **Lower** MSE means **better** predictions.
- **R^2 Score**: How much of the price variation is explained by the model. **1.0** = perfect prediction, **0** = no predictive power.

---

### 🖼️ Step 7: Visualize the Regression Line

Let's plot the actual vs predicted data to see how well our model fits.

```python
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price")
plt.title("House Size vs Price")
plt.legend()
plt.grid(True)
plt.show()
```

Explanation:

- **Blue dots** show actual house prices from test data.
- The **red line** is the model’s prediction.
- A good model will have points close to the **red line**.

---

## 📊 Step 8: Prediction on New Data

You can also use the trained pipeline to predict outcomes for completely new input - e.g., for new house sizes not in your dataset.

Here's how to create a new data sample and predict their scores:

```python
# Create new house size data for prediction
new_house_sizes = pd.DataFrame({
    "House_Size": [720, 780, 860, 930, 990, 1030, 1080, 1120, 1150, 1200]
})

# Predict using the trained model
new_predictions = model.predict(new_house_sizes)

# Print predicted prices
print("\nPredictions for New House Sizes:")
for size, price in zip(new_house_sizes["House_Size"], new_predictions):
    print(f"House Size: {size} sq ft -> Predicted Price: ${price:,.2f}")

# Plot training/test data
plt.scatter(X, y, color='lightgray', label='Original Data')

# Plot regression line based on full data
full_line = model.predict(X)
plt.plot(X, full_line, color='red', linewidth=2, label='Regression Line')

# Plot new prediction points
plt.scatter(new_house_sizes, new_predictions, color='green', label='New Predictions', marker='x', s=80)

plt.xlabel("House Size (sq ft)")
plt.ylabel("Price")
plt.title("Linear Regression - New Predictions")
plt.legend()
plt.grid(True)
plt.show()
```

**Explanation:**

- New data (`new_house_sizes`) contains unseen house sizes to test generalization.
- The model uses `.predict()` to estimate their prices.
- Visualization:
  - Original data in **gray**.
  - Regression line in **red**.
  - New predictions marked with **green 'X'**.

---

## 🧪 Practice Exercises

### ✅ Exercise 1: Try Your Own Values 🛠️

One of the best ways to understand how Linear Regression works is by **experimenting**. In this exercise, you'll:

- Modify the house size values.
- Retrain the model.
- Observe how predictions and accuracy change.

This helps you see how data directly affects the model's learning and performance.

Let's simulate a new dataset with different house sizes and prices.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a new dataset with your own values
new_data = pd.DataFrame({
    "House_Size": [600, 850, 1200, 1500, 1800, 2000, 2400, 2800],
    "Price":      [150000, 180000, 240000, 300000, 330000, 360000, 420000, 470000]
})

# Step 2: Split the data
X = new_data[["House_Size"]]
y = new_data["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 3: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Predicted Prices:", y_pred)
print("Mean Squared Error:", mse)
print("R-squared (R^2):", r2)

# Step 6: Visualize the results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price")
plt.title("House Size vs Price (New Data)")
plt.legend()
plt.grid(True)
plt.show()
```

We can test the model on completely new data (8-10 house sizes) and visualize those predictions.

```python
# Step 6: Predict on completely new house sizes
new_house_sizes = pd.DataFrame({
    "House_Size": [650, 900, 1300, 1600, 1900, 2100, 2500, 3000]
})

new_predictions = model.predict(new_house_sizes)

# Print new predictions
print("\nPredictions for New House Sizes:")
for size, price in zip(new_house_sizes["House_Size"], new_predictions):
    print(f"House Size: {size} sq ft -> Predicted Price: ${price:,.2f}")

# Step 7: Visualize the data and predictions
plt.figure(figsize=(10, 6))

# Plot training/testing data
plt.scatter(X, y, color='lightgray', label='Original Data')

# Plot regression line
line = model.predict(X)
plt.plot(X, line, color='red', linewidth=2, label='Regression Line')

# Plot new prediction points
plt.scatter(new_house_sizes, new_predictions, color='green', marker='x', s=100, label='New Predictions')

plt.xlabel("House Size (sq ft)")
plt.ylabel("Price")
plt.title("House Size vs Price - New Predictions")
plt.legend()
plt.grid(True)
plt.show()
```

**Explanation:**

- `new_house_sizes`: 8 new house sizes to test your trained model
- `model.predict(new_house_sizes)`: predict prices
- Visual plot includes:

  - Original data (**gray**)
  - Regression line (**red**)
  - New prediction points (**green Xs**)

---

#### 🧠 What You'll Learn from This Exercise:

- How model predictions change when you give it new data.
- How different data patterns affect the **slope**, **intercept**, and **R² score**.
- How model visualization helps in spotting trends and accuracy.

---

### 🧮 Exercise 2: Add a New Feature (Multiple Linear Regression)

So far, we've only used **one feature** `House_Size` to predict house prices. But in real life, price can depend on many things, like:

- 📏 Size of the house
- 🛏️ Number of bedrooms
- 📍 Location
- 🛁 Number of bathrooms

In this exercise, we'll **add a new feature**: `Bedrooms` to see how a multiple linear regression model works.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create dataset with 2 features
data = pd.DataFrame({
    "House_Size": [750, 800, 850, 900, 950, 1000, 1050, 1100],
    "Bedrooms":   [2,   2,   2,   3,   3,    3,    4,    4],
    "Price":      [150000, 160000, 165000, 175000, 180000, 190000, 200000, 210000]
})

# Step 2: Split features (X) and target (y)
X = data[["House_Size", "Bedrooms"]]  # Note: 2 features now
y = data["Price"]

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Predicted Prices:", y_pred)
print("Mean Squared Error:", mse)
print("R-squared (R^2):", r2)

# Step 7: View learned parameters
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Step 8: Visualize Predictions vs Actual Prices
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='green', edgecolor='black')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line (perfect prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()
```

**Explanation:**

- **Green dots**: Actual vs predicted price points.
- **Red dashed line**: Ideal line where predicted = actual.
- If all points lie on the red line, the model is perfect.
- The **closer the dots are to the red line**, the better the model.

This plot is useful even with multiple features because it compares **output**, not input dimensions.

Let's create a new dataset with 8-10 entries of `House_Size` and `Bedrooms`. Predict prices using your trained model and visualize those predictions

```python
# Step 9: Predict on new data (not in training/testing)
new_houses = pd.DataFrame({
    "House_Size": [720, 780, 840, 910, 970, 1020, 1080, 1130, 1180, 1250],
    "Bedrooms":   [2,   2,   2,   3,   3,    3,    4,    4,    4,    5]
})

# Predict prices for new houses
new_predictions = model.predict(new_houses)

# Display predictions
print("\nPredictions for New Houses:")
for size, beds, price in zip(new_houses["House_Size"], new_houses["Bedrooms"], new_predictions):
    print(f"House Size: {size} sq ft | Bedrooms: {beds} -> Predicted Price: ${price:,.2f}")

# Step 10: Visualize new predictions
plt.figure(figsize=(9, 6))

# Plot original data (actual)
plt.scatter(data["House_Size"], data["Price"], color='lightgray', label='Training Data')

# Plot new predicted points
plt.scatter(new_houses["House_Size"], new_predictions, color='blue', marker='x', s=100, label='New Predictions')

# Label the plot
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price")
plt.title("House Size vs Predicted Price (New Data)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

**Explanation**:

- `new_houses`: Contains 10 new examples, including variation in both house size and number of bedrooms.
`model.predict(new_houses)`: Uses your trained model to estimate prices for these unseen combinations.
- Visualization:

  - **Gray dots**: original training data
  - **Blue Xs**: predictions on new data (what your model has learned to generalize)

---

#### 🧠 What's Different Here?

- `X` now contains **two columns**: `House_Size` and `Bedrooms`.
- The model learns **two coefficients** instead of one.
- This becomes a **multiple linear regression** model:

```
Price = a1 * House_Size + a2 * Bedrooms + b
```

Where:

- `a1` is the coefficient for `House_Size`
- `a2` is the coefficient for `Bedrooms`
- `b` is the intercept

---

### 💾 Exercise 3: Save & Load Model using Joblib

Once you've trained your linear regression model, you can **save it to a file** so you don't have to retrain it every time. This is useful when you're building real-world applications where loading a pre-trained model is faster and more efficient.

We'll use Python's `joblib` library to save the trained model to a file (`.pkl` format) and then load it later to make predictions.

---

#### 📦 What is joblib?

- `joblib` is a utility from Scikit-learn used to save models and pipelines efficiently.
- It saves not just the model's structure but also its trained parameters. 

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # New import

# Step 1: Create dataset
data = pd.DataFrame({
    "House_Size": [750, 800, 850, 900, 950, 1000, 1050, 1100],
    "Bedrooms":   [2,   2,   2,   3,   3,    3,    4,    4],
    "Price":      [150000, 160000, 165000, 175000, 180000, 190000, 200000, 210000]
})

# Step 2: Split features and target
X = data[["House_Size", "Bedrooms"]]
y = data["Price"]

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Save the trained model to disk
joblib.dump(model, "house_price_model.pkl")
print("Model saved as 'house_price_model.pkl'")

# Step 6: Load the saved model from disk
loaded_model = joblib.load("house_price_model.pkl")
print("Model loaded successfully")

# Step 7: Use the loaded model to make predictions
sample_input = pd.DataFrame([[950, 3]], columns=["House_Size", "Bedrooms"])  # Example: house with 950 sqft and 3 bedrooms
predicted_price = loaded_model.predict(sample_input)

print("Prediction from loaded model for 950 sqft & 3 bedrooms:", predicted_price[0])
```

**Explanation:**

- `joblib.dump(model, "filename.pkl")`: Saves the trained model to a `.pkl` file.
- `joblib.load("filename.pkl")`: Loads the model back into memory.
- You can now **make predictions without retraining the model**, even if the program was restarted or deployed elsewhere.

This is especially useful for **deploying machine learning models into real-world apps**, such as:

- Predicting house prices on a website.
- Integrating with APIs or mobile apps.
- Reusing trained models in future projects

---

## 🧠 Summary

| Task                    | Code/Function                    |
| ----------------------- | -------------------------------- |
| Train-test split        | `train_test_split()`             |
| Train Linear Regression | `model.fit(X_train, y_train)`    |
| Predict                 | `model.predict(X_test)`          |
| Evaluate MSE, R²        | `mean_squared_error`, `r2_score` |
| Visualize results       | `matplotlib.pyplot`              |

---

## 🔗 Related Files

 - [Day17_LinearRegression.py](../Day17_LinearRegression/Day17_LinearRegression/Day17_LinearRegression.py) - Code example
  - [PracticeExercise1.py](../Day17_LinearRegression/Day17_LinearRegression/PracticeExercise1.py) - Practice Exercise 1 code example
  - [PracticeExercise2.py](../Day17_LinearRegression/Day17_LinearRegression/PracticeExercise2.py) - Practice Exercise 2 code example
  - [PracticeExercise3.py](../Day17_LinearRegression/Day17_LinearRegression/PracticeExercise3.py) - Practice Exercise 3 code example  

📅 Up next: **Day 18 - Logistic Regression & Classification Metrics** - Learn how to use logistic regression for binary classification problems like predicting spam vs not-spam or pass vs fail.