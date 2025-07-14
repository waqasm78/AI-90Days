
import pandas as pd

# Load sample data
df = pd.read_csv("students.csv")
print("Original Data:")
print(df)

# Remove duplicates and invalid entries
df = df.drop_duplicates()
df = df[df["Age"] >= 0]
print("\nAfter Removing Duplicates and Invalid Age:")
print(df)

# Handle missing values
print("\nMissing values before handling:")
print(df.isnull().sum())
df["Score"] = df["Score"].fillna(df["Score"].mean())
print("\nAfter Filling Missing Scores:")
print(df)

# Clean and standardize text data
df["City"] = df["City"].str.lower().str.strip()
df["City"] = df["City"].replace({"lahor": "lahore"})
print("\nAfter Cleaning City Names:")
print(df)

# Encode categorical variables
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
print("\nAfter Encoding Gender:")
print(df)

# Feature engineering: Result
df["Result"] = df["Score"].apply(lambda x: "Pass" if x >= 50 else "Fail")
print("\nAfter Adding Result Column:")
print(df)

# Binning score into performance levels
bins = [0, 50, 70, 100]
labels = ["Low", "Medium", "High"]
df["Performance"] = pd.cut(df["Score"], bins=bins, labels=labels)
print("\nAfter Binning Score into Performance:")
print(df)

# Practice Exercise 1: Fill missing Age
missing_age_count = df["Age"].isnull().sum()
print("\nMissing Age Count:", missing_age_count)
df["Age"] = df["Age"].fillna(df["Age"].median())
print("\nAfter Filling Missing Age:")
print(df)

# Practice Exercise 2: Create Age Groups
age_bins = [0, 18, 25, 100]
age_labels = ["Teen", "Young Adult", "Adult"]
df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels)
print("\nAfter Creating Age Groups:")
print(df[["Name", "Age", "AgeGroup"]])

# Practice Exercise 3: Normalize Scores
df["Score_Norm"] = (df["Score"] - df["Score"].min()) / (df["Score"].max() - df["Score"].min())
print("\nAfter Normalizing Scores:")
print(df[["Name", "Score", "Score_Norm"]])

# Save cleaned data
df.to_csv("students_cleaned.csv", index=False)
