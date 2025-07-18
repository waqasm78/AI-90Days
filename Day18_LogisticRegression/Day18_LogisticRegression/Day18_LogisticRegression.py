import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Sample Data
students = pd.DataFrame({
    "Study_Hours": [2, 4, 5, 6, 8, 9, 10, 11],
    "Attendance": [40, 50, 52, 60, 85, 87, 90, 95],
    "Passed":     [0, 0, 0, 0, 1, 1, 1, 1]
})

# Step 2: Split Features and Target
X = students[["Study_Hours", "Attendance"]]
y = students["Passed"]

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Step 4: Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Predictions
predictions = model.predict(X_test)

# Step 6: Evaluate
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))