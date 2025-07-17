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
