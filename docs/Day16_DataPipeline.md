# Day 16 - Data Preprocessing and Pipelines in Scikit-learn 🔄🧪

Welcome to **Day 16** of your AI-90Days journey! Today you'll learn about **pipelines** in Scikit-learn - an essential tool to streamline and automate the process of data preprocessing and machine learning model training. Pipelines help avoid data leakage and make your code more modular and reproducible.

---

## 🌟 Objectives

- Understand what data preprocessing and pipelines are and why they are useful
- Learn why pipelines are important for reproducibility and efficiency and how to build a pipeline using Scikit-learn
- Combine preprocessing and modeling steps into one streamlined workflow
- Make predictions using a trained pipeline
- Build your first preprocessing pipeline in Scikit-learn
- Practice using real-world examples

---

## 🔍 What is Data Preprocessing?

**Data preprocessing** is the transformation of raw data into a clean and usable format. This may include:

- Filling missing values
- Encoding categorical features
- Scaling numerical features
- Removing duplicates
- Normalizing or standardizing features

ML models cannot learn well from raw, messy data, preprocessing ensures all features are in the right format for modeling.

---

## ⚙️ What Are Pipelines?

A **pipeline** in machine learning is a way to bundle all the preprocessing and modeling steps into a single object. This makes it easy to train, test, and reuse the model without repeating the entire preprocessing logic every time.

A **pipeline** is a tool in **Scikit-learn** that chains multiple steps (like preprocessing and modeling) together so they can be executed sequentially. Pipelines help:

- Reduce repetitive code
- Improve model reproducibility
- Prevent data leakage (leaking test data into training)
- Simplify deployment

---

## 1️⃣ Import Required Libraries

Before we start building a machine learning pipeline, we need to import all the essential libraries that will help us with data preprocessing, modeling, and pipeline creation.

These libraries are part of the **Scikit-learn** and **Pandas** ecosystem, which are widely used in machine learning and data science projects. Each of these tools serves a specific purpose in cleaning, transforming, and modeling the data.


```python
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
```

**Explanation:**

- `pandas` - Used for loading, exploring, and manipulating structured data (especially CSV files and DataFrames).
- `train_test_split` - Splits your dataset into training and testing sets, which helps in evaluating your model properly.
- `StandardScaler` - Standardizes numeric features by removing the mean and scaling to unit variance. This is essential for algorithms that are sensitive to feature scale.
- `OneHotEncoder` - Converts categorical (non-numeric) data into a format that can be provided to ML models (binary format).
- `Pipeline` - Bundles preprocessing steps and model training into a single object, making the code cleaner and less error-prone.
- `ColumnTransformer` - Applies different preprocessing transformations to different columns (e.g., scale numbers and encode categories separately).
- `SimpleImputer` - Handles missing values by replacing them with strategies like mean, median, or a constant value.
- `LinearRegression` - A simple and widely used regression model that tries to find a linear relationship between input features and the target.
- `DecisionTreeRegressor` - A machine learning model that makes predictions by learning decision rules inferred from the data (alternative to linear regression).
- `joblib` - Used for saving and loading trained machine learning models (persistence).

---

## 2️⃣ Load Dataset

Before we build a machine learning pipeline, we need some data to work with. For this example, we're going to simulate a small dataset representing students' study habits and performance.

This dataset contains:

- A **categorical** feature: `Gender`
- Two **numeric** features: `Hours_Studied` and `Attendance`
- A **target/output**: `Score` (the exam result we want to predict)

This kind of simple data is perfect for understanding the mechanics of preprocessing and model training using pipelines.

```python
# Sample dataset
data = pd.DataFrame({
    "Gender": ["Male", "Female", "Female", "Male", "Female"],
    "Hours_Studied": [5, 8, 7, 4, 6],
    "Attendance": [90, 95, 92, 85, 88],
    "Score": [75, 88, 85, 60, 80]
})

print(data)
```

**Explanation:**

We simulate a DataFrame of 5 students. Each student has:

- A `Gender`
- How many `Hours_Studied`
- Their `Attendance` percentage
- Their final `Score`

This simple, mixed-type data allows us to apply both numeric and categorical preprocessing.

---

## 3️⃣ Split Data for Training and Testing

In machine learning, we never train a model on the full dataset. Instead, we split it into two parts:

- A **training set** to teach the model
- A **testing set** to evaluate how well the model performs on unseen data

We'll use `train_test_split()` to separate our features and target, then divide them into training and test sets and predict `Score` based on the other features.

```python
X = data.drop("Score", axis=1)
y = data["Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Explanation:**

We split data so that we train the model on one part and evaluate it on another.

- `X` contains all input columns (features) except the target.
- `y` is the column we want to predict: Score.
- `train_test_split` splits 80% of the data into training, and 20% into testing.
- `random_state=42` ensures reproducibility so the split is always the same.

---

## 4️⃣ Create a Preprocessing Pipeline

Different types of data need different kinds of preprocessing:

- Numeric data (like `Hours_Studied`) should be **scaled** so that large values don't dominate.
- Categorical data (like `Gender`) should be **encoded** into numeric form using one-hot encoding.

We will define separate pipelines for numeric and categorical columns, and combine them using `ColumnTransformer`.

```python
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
```

**Explanation:**

- `numeric_cols`: These are columns that will be standardized using `StandardScaler`.
- `categorical_cols`: These will be transformed using `OneHotEncoder`, which converts each category (like "Male"/"Female") into separate columns of 1s and 0s.
- `ColumnTransformer`: This applies the correct transformation pipeline to the respective column type.

By using this setup, we ensure that both kinds of data are properly prepared before feeding them into our machine learning model.

---

## 5️⃣ Build Final Pipeline (Preprocessing + Model)

Once the data is prepared using a preprocessor, the next step is to combine everything into one clean workflow using a **Pipeline**.

A pipeline allows you to **chain preprocessing and modeling steps together**, making the process simpler, reusable, and less error-prone.

```python
# Full pipeline
model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", LinearRegression())
])
```
**Explanation:**

We create a complete pipeline that first preprocesses the data and then fits a regression model - all in one step.

- `"preprocessing"`: This step handles both scaling and encoding based on our `ColumnTransformer`.
- `"regressor"`: The actual model that will be trained. We're using `LinearRegression` here.

Now, instead of manually transforming the data and fitting the model, we can just fit this pipeline - and it will take care of everything automatically.

---

## 6️⃣ Train the Pipeline

Once your preprocessing and modeling steps are combined into a single pipeline, the next step is to train (or fit) it on your training data. This means:

- The preprocessing steps (scaling, encoding) will be applied to your `X_train`
- The model (like `LinearRegression`) will then be trained on this transformed data

Let's now fit the pipeline to the training data.

```python
model_pipeline.fit(X_train, y_train)
```

**Explanation:**

This step trains the scaler and model on training data - all inside the pipeline.

- `fit()` is the key function in Scikit-learn that both transforms the input data (using the pipeline) and trains the model.
- The pipeline ensures all transformations are done only on training data, avoiding data leakage.

---

## 7️⃣ Predict on Test Data

Now that your model is trained, you can use it to make predictions on new, unseen data (like your test set).

The pipeline will:

- Automatically apply the **same preprocessing steps** to `X_test`
- Use the **trained model** to generate predictions

We now use the trained pipeline to predict the scores for test data.

```python
y_pred = model_pipeline.predict(X_test)
print("Predicted Scores on Test Set:", y_pred)
```

**Explanation:**

All test data will be scaled and predicted in one step automatically.

- You don't need to preprocess `X_test` manually - the pipeline handles that for you.
- This ensures **consistency** between training and prediction, which is crucial for accurate results.

---

## 8️⃣ Predict on New Data

You can also use the trained pipeline to predict outcomes for completely new input - e.g., for new students not in your dataset.

Here's how to create a new data sample and predict their scores:

```python
# New unseen data
new_students = pd.DataFrame({
    "Gender": ["Female", "Male"],
    "Hours_Studied": [6, 3],
    "Attendance": [94, 80]
})

# Predict using trained pipeline
new_predictions = model_pipeline.predict(new_students)
print("Predicted Scores for New Students:", new_predictions)
```

**Explanation:**

You don't need to manually scale the new input - pipeline does it all.

- `new_students` is a **DataFrame** with the same column names and structure as the original training data.
- You can pass it directly into the pipeline and it will automatically:

  -  Encode `"Gender"`
  -  Scale `Hours_Studied` and `Attendance`
  -  Use the trained model to generate predictions

---

## 🧪 Practice Exercises

### ✅ Practice Exercise 1: Add Missing Values and Handle Them

Let's modify the dataset to include missing values.

```python
data_missing = pd.DataFrame({
    "Gender": ["Male", "Female", None, "Male", "Female"],
    "Hours_Studied": [5, 8, 7, None, 6],
    "Attendance": [90, 95, 92, 85, None],
    "Score": [75, 88, 85, 60, 80]
})

X = data_missing.drop("Score", axis=1)
y = data_missing["Score"]
```

Now update your preprocessing pipeline:

```python
from sklearn.impute import SimpleImputer

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
```

**Explanation:**

- `SimpleImputer()` fills in missing values.
- For numbers: use mean
- For categories: use the most common value

---

### ✅ Practice Exercise 2: Change the Model to Decision Tree

Try swapping the model from linear regression to decision tree:

```python
model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", DecisionTreeRegressor())
])

model_pipeline.fit(X_train, y_train)
```

**Explanation:**

Pipelines are flexible! You can plug in different models easily without touching the rest of the code.

---

### ✅ Practice Exercise 3: Save and Reload the Pipeline

Use joblib to save the pipeline to disk and reload it later.

```python
# Save
joblib.dump(model_pipeline, "student_model.pkl")

# Load
loaded_model = joblib.load("student_model.pkl")

# Predict
print("Reloaded model prediction:", loaded_model.predict(X_test))
```

**Explanation:**

Saving your pipeline allows you to reuse it without retraining - great for production!

---

### 🎁 Bonus: Classification Pipeline - Predicting Pass or Fail 🧠✔️❌

So far, we've been predicting continuous numeric values (like scores). But many real-world problems involve **categorical predictions**, like:

- Will a student **Pass** or **Fail**?
- Will a customer **buy** or **not buy**?
- Is this email **Spam** or **Not Spam**?

This type of prediction is called classification. We'll now build a full pipeline to predict whether a student **Passes** or **Fails**, using the same approach we used for regression, but this time using **Logistic Regression**, which is a classifier.

```python
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
```

**Explanation:**

- **Classification vs Regression**: We use `LogisticRegression()` instead of `LinearRegression()` because our target (`Result`) is **categorical**, not numeric.
- **Why use Pipelines again?**: Pipelines allow us to **preprocess and train** in one step. We can even reuse this structure in real-world deployment.
- **OneHotEncoder for Gender**: Converts `"Male"` and `"Female"` into separate binary columns so the model can interpret the data.
- **Prediction**: The model outputs a label (`Pass` or `Fail`) based on study hours, attendance, and gender.

---

## 🔗 Related Files

- [Day16_DataPipeline.py](../Day16_DataPipeline/Day16_DataPipeline.py) - All examples predicting continuous numeric values
- [Day16_DataPipeline1.py](../Day16_DataPipeline/Day16_DataPipeline1.py) - Example predicting categorical values like **Pass** or **Fail**

---

## 🧠 Summary

| Task                        | Method Used                      |
|-----------------------------|----------------------------------|
| Split dataset               | `train_test_split()`             |
| Standardize features        | `StandardScaler()`               |
| Train regression model      | `LinearRegression()`             |
| Combine steps               | `Pipeline()`                     |
| Predict future values       | `pipeline.predict(new_data)`     |

📅 Up next: **Day 17 - Supervised Learning: Classification & Regression** - Learn how to build real prediction models using labeled data!