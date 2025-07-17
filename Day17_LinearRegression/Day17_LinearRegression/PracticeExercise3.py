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