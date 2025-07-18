import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_curve, auc
)

# Step 1: Load Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic = pd.read_csv(url)

# Step 2: Feature Engineering
# Create a new column: Title extracted from Name (Mr, Mrs, Miss, etc.)
titanic["Title"] = titanic["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

# Simplify rare titles
rare_titles = titanic["Title"].value_counts()[titanic["Title"].value_counts() < 10].index
titanic["Title"] = titanic["Title"].replace(rare_titles, "Rare")

# Fill missing Embarked with most common
titanic["Embarked"] = titanic["Embarked"].fillna(titanic["Embarked"].mode()[0])

# Drop Cabin and Ticket (too many missing values or non-numeric)
titanic = titanic.drop(columns=["Cabin", "Ticket", "Name", "PassengerId"])

# Encode categorical features
titanic["Sex"] = titanic["Sex"].map({"male": 0, "female": 1})
titanic["Embarked"] = titanic["Embarked"].map({"S": 0, "C": 1, "Q": 2})
titanic["Title"] = titanic["Title"].map({
    "Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4
})

# Fill missing Age
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# Select Features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title"]
X = titanic[features]
y = titanic["Survived"]

print(features)
for test_size in [0.1, 0.2, 0.3]:
    print(f"\nTest Size: {test_size}")
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train Model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = auc(*roc_curve(y_test, y_probs)[:2])
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"AUC Score: {roc_auc:.3f}")

# Show first 10 passengers' survival probabilities
probs = model.predict_proba(X_test)
predictions_df = pd.DataFrame({
    "Survival_Probability": probs[:, 1],
    "Predicted_Label": model.predict(X_test),
    "Actual_Label": y_test.reset_index(drop=True)
})
print("\nFirst 10 Predictions with Probabilities:\n", predictions_df.head(10))

# Optional: Visualize coefficients (importance)
coeffs = pd.Series(model.coef_[0], index=features)
coeffs.sort_values().plot(kind='barh', title="Feature Importance (Logistic Coefficients)")
plt.tight_layout()
plt.grid(True)
plt.show()