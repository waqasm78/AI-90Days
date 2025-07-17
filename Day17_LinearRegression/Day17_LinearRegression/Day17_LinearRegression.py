# Day 17 - Linear Regression

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

# Separate features (X) and target (y)
X = data[["House_Size"]]  # Feature column (must be 2D)
y = data["Price"]         # Target column

# Split the data into training (75%) and testing (25%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create the model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# View Learned Parameters
print("Slope (Coefficient):", model.coef_)
print("Intercept:", model.intercept_)

# Make predictions on the test set
y_pred = model.predict(X_test)
print("Predicted Prices:", y_pred)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared (R^2):", r2)

# Visualize the results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price")
plt.title("House Size vs Price")
plt.legend()
plt.grid(True)
plt.show()


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