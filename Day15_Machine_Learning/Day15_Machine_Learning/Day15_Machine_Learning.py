# Day 15 - Introduction to Machine Learning with Scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

# --------------------------------------
# 1️ Import Libraries and Prepare Data
# --------------------------------------
data = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6],
    "Score": [35, 45, 50, 60, 68, 75]
})
print("Original Small Dataset:\n", data)

X = data[["Hours"]]  # Input (2D)
y = data["Score"]    # Output

# --------------------------------------
# 2️ Split Data for Training and Testing
# --------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------
# 3️ Train the Model
# --------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------
# 4️ Make Predictions
# --------------------------------------
predictions = model.predict(X_test)
print("\nPredicted Scores:", predictions)

# New data (same structure as training data)
new_data = pd.DataFrame({
    "Hours": [1.5, 3.5, 5.5, 7]   # You can test any values
})

# Predict the scores
predicted_scores = model.predict(new_data)

# Print results
print("\nPredicted Scores:", predicted_scores)

# --------------------------------------
# 5️ Evaluate the Model
# --------------------------------------
print("\nModel Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R² Score:", r2_score(y_test, predictions))

# --------------------------------------
# Practice Exercise 1: Full Pipeline with MAE
# --------------------------------------
df = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Score": [32, 45, 50, 60, 65, 72, 80, 90]
})
print("\nExtended Dataset:\n", df)

X = df[["Hours"]]
y = df["Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print("\nMean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))

# --------------------------------------
# Practice Exercise 2: Visualize the Regression Line
# --------------------------------------
plt.figure(figsize=(6, 4))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, reg.predict(X), color="red", label="Regression Line")
plt.title("Study Hours vs Score")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------
# Practice Exercise 3: Predict New Value
# --------------------------------------
new_hours = [[9.5]]
new_prediction = reg.predict(new_hours)
print("\nPredicted Score for 9.5 study hours:", new_prediction[0])

