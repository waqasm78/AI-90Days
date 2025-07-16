# Day 16 – Data Preprocessing and Pipelines in Scikit-learn 

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

# Sample dataset
data = pd.DataFrame({
    "Gender": ["Male", "Female", "Female", "Male", "Female"],
    "Hours_Studied": [5, 8, 7, 4, 6],
    "Attendance": [90, 95, 92, 85, 88],
    "Score": [75, 88, 85, 60, 80]
})

print(data)

# Splitting the dataset for Training and Testing

X = data.drop("Score", axis=1)
y = data["Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Convert categorical data to numerical
# Define which columns are categorical and numeric

categorical_cols = ["Gender"]
numeric_cols = ["Hours_Studied", "Attendance"]

# Pipeline for numeric features

numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

# Pipeline for categorical features

categorical_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both pipelines

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# Full pipeline

model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", LinearRegression())
])

# Fit the model: Train the Pipeline

model_pipeline.fit(X_train, y_train)

# Evaluate the model: Score on the test set
# Predict on Test Data

y_pred = model_pipeline.predict(X_test)
print("Predicted Scores:", y_pred)

# New unseen data
new_students = pd.DataFrame({
    "Gender": ["Female", "Male"],
    "Hours_Studied": [6, 3],
    "Attendance": [94, 80]
})

# Predict using trained pipeline
new_predictions = model_pipeline.predict(new_students)
print("Predicted Scores for New Students:", new_predictions)

# New data for prediction
# Example: A Female student who studied 6.5 hours and had 93% attendance
new_data = pd.DataFrame({
    "Gender": ["Female"],
    "Hours_Studied": [6.5],
    "Attendance": [93]
})

# Predict the score using the trained pipeline
new_pred = model_pipeline.predict(new_data)
print("Predicted Score for new data:", new_pred[0])

# Practice Exercise 1: Add Missing Values and Handle Them

data_missing = pd.DataFrame({
    "Gender": ["Male", "Female", None, "Male", "Female"],
    "Hours_Studied": [5, 8, 7, None, 6],
    "Attendance": [90, 95, 92, 85, None],
    "Score": [75, 88, 85, 60, 80]
})

X = data_missing.drop("Score", axis=1)
y = data_missing["Score"]

# Numeric pipeline with imputer
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Categorical pipeline with imputer
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Practice Exercise 2: Change the Model to Decision Tree

model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", DecisionTreeRegressor())
])

model_pipeline.fit(X_train, y_train)

# Practice Exercise 3: Save and Reload the Pipeline

# Save
joblib.dump(model_pipeline, "student_model.pkl")

# Load
loaded_model = joblib.load("student_model.pkl")

# Predict
print("Reloaded model prediction:", loaded_model.predict(X_test))
