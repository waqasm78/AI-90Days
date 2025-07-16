# Day 15 – Introduction to Machine Learning with Scikit-learn

# Step 1: Load the Dataset
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# Load Iris Dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target
print("First 5 rows of Iris Dataset:")
print(df.head())

# Step 2: Split Data
X = df[iris.feature_names]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Model using KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 4: Make Predictions
predictions = model.predict(X_test)
print("\nPredicted Labels:")
print(predictions)

# Step 5: Evaluate Model
accuracy = accuracy_score(y_test, predictions)
print("KNN Accuracy:", accuracy)

# ================================
# Practice Exercise 1: Decision Tree Classifier
# ================================
print("\nTraining Decision Tree Classifier")
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)
tree_acc = accuracy_score(y_test, tree_preds)
print("Decision Tree Accuracy:", tree_acc)

# ================================
# Practice Exercise 2: Breast Cancer Dataset
# ================================
print("\nLoading Breast Cancer Dataset")
cancer_data = load_breast_cancer()
df_cancer = pd.DataFrame(data=cancer_data.data, columns=cancer_data.feature_names)
df_cancer["target"] = cancer_data.target
print("First 5 rows of Cancer Dataset:")
print(df_cancer.head())

# Prepare Cancer Data
X_cancer = df_cancer[cancer_data.feature_names]
y_cancer = df_cancer["target"]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cancer, y_cancer, test_size=0.3, random_state=0)

# Train KNN on Cancer Dataset
cancer_model = KNeighborsClassifier()
cancer_model.fit(Xc_train, yc_train)
cancer_preds = cancer_model.predict(Xc_test)
cancer_acc = accuracy_score(yc_test, cancer_preds)
print("Cancer Dataset Accuracy:", cancer_acc)

# ================================
# Practice Exercise 3: Tune k in KNN
# ================================
print("\nTrying Different k Values for KNN")
for k in range(1, 6):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    print(f"K={k} -> Accuracy: {acc:.2f}")
