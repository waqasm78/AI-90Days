# Day 16 - Data Preprocessing and Pipelines in Scikit-learn 

# Introduction:
# This script demonstrates how to use preprocessing tools and pipelines in Scikit-learn to streamline machine learning workflows.
# Pipelines ensure that all data transformations and modeling steps are applied consistently and correctly.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1️ Sample Dataset
print("\nSample Dataset:")
data = pd.DataFrame({
    "Hours_Studied": [1, 2, 3, 4, 5, 6],
    "City": ["Lahore", "Karachi", "Lahore", "Islamabad", "Karachi", "Lahore"],
    "Score": [35, 45, 50, 60, 68, 75]
})
print(data)

# 2️ Features and Target
print("\nFeatures and Target:")
X = data[["Hours_Studied", "City"]]
y = data["Score"]
print("Features (X):\n", X)
print("Target (y):\n", y)

# 3️ Preprocessing Pipelines for Numeric and Categorical Data
print("\nColumn Transformers:")
numeric_features = ["Hours_Studied"]
categorical_features = ["City"]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 4️ Create a Full Pipeline with Preprocessing + Model
print("\nFull Pipeline Creation:")
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])
print("Pipeline created successfully.")

# 5️ Split Data into Train and Test
print("\nSplitting Data:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Training Data X:")
print(X_train)
print("Testing Data X:")
print(X_test)
print("Training Data Y:")
print(y_train)
print("Testing Data Y:")
print(y_test)

# 6️ Train the Pipeline
print("\nTraining the Pipeline:")
model.fit(X_train, y_train)
print("Model training complete.")

# 7️ Predict on Test Data
print("\nPredict on Test Data:")
y_pred = model.predict(X_test)
print("Predictions:", y_pred)

# 8️Evaluate the Model
print("\nEvaluate the Model:")
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# 9️ Predict on New Data
print("\nPredict on New Data:")
new_data = pd.DataFrame({
    "Hours_Studied": [2, 5],
    "City": ["Lahore", "Islamabad"]
})
new_predictions = model.predict(new_data)
print("New Predictions:", new_predictions)
