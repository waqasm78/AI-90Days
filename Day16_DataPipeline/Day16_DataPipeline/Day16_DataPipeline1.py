import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Step 1: Create dataset
data = pd.DataFrame({
    "Gender": ["Male", "Female", "Female", "Male", "Female"],
    "Hours_Studied": [5, 8, 7, 4, 6],
    "Attendance": [90, 95, 92, 85, 88],
    "Score": [75, 88, 85, 60, 80]
})

# Step 2: Convert score to categorical outcome
data["Result"] = data["Score"].apply(lambda x: "Pass" if x >= 70 else "Fail")
data = data.drop("Score", axis=1)

# Step 3: Define features and target
X = data.drop("Result", axis=1)
y = data["Result"]

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Preprocessing for numeric and categorical
numeric_cols = ["Hours_Studied", "Attendance"]
categorical_cols = ["Gender"]

numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# Step 6: Final pipeline with Logistic Regression
clf_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression())
])

# Step 7: Train the pipeline
clf_pipeline.fit(X_train, y_train)

# Step 8: Predict on new data
new_data = pd.DataFrame({
    "Gender": ["Female", "Male"],
    "Hours_Studied": [6, 3],
    "Attendance": [92, 80]
})

predictions = clf_pipeline.predict(new_data)

# Show predictions
print("Predicted Results for New Data:", predictions)
