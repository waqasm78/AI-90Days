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
