# Day 9 - Working with CSV and JSON Files 📊📦

Welcome to **Day 9**! Today, we'll explore how to handle structured data formats - **CSV (Comma-Separated Values) and JSON (JavaScript Object Notation)**. These are two of the most common formats used in real-world AI projects to store and exchange data, from training datasets to APIs.

---

## 🌟 Objectives

- Understand the structure and use of CSV and JSON files
- Learn how to read and write CSV files using csv module
- Learn how to parse and modify JSON data using json module
- Use Visual Studio 2022 or Jupyter Notebook to practice

---

## 🔍 Why CSV and JSON Matter?

Most real-world data comes from spreadsheets (CSV) or APIs (JSON). To train models, evaluate results, or automate decisions, you'll often load these formats into memory.

Examples:

 - **CSV:** "sales.csv", "iris.csv", "students.csv"
 - **JSON:** "user_profile.json", "settings.json", or REST API responses

---

## 📑 Reading CSV Files

CSV files store data in rows and columns. Python's `csv` module makes it easy to work with this format.

### 📘 Code Example: Read a CSV File

```python
import csv

with open("students.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)
```

#### 📖 Explanation

 - `csv.reader(file)` reads each row as a list of values
 - `for row in reader:` lets you process the file row-by-row
 - Useful for datasets saved in spreadsheet format

---

## ✍️ Writing CSV Files

You can save data to CSV using `csv.writer()` which writes rows as comma-separated values.

### 📘 Code Example: Write to a CSV File

```python
import csv

with open("output.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Age"])
    writer.writerow(["Ali", 21])
    writer.writerow(["Sara", 22])
```

#### 📖 Explanation

 - `writerow([...])` writes a list as one row
 - `newline=''` ensures there's no extra blank line between rows (important on Windows)
 - Great for saving structured tabular data

---

## 🔄 Reading JSON Files

JSON is a hierarchical data format used widely in APIs, configuration files, and machine learning data.

### 📘 Code Example: Read a JSON File

```python
import json

with open("data.json", "r") as file:
    data = json.load(file)
    print(data)
```

#### 📖 Explanation

 - `json.load()` loads the entire file as a Python dictionary
 - Ideal for nested data (e.g., objects inside arrays)
 - Allows direct access using keys, e.g., `data["name"]`

✍️ Writing JSON Files

To save data in JSON format, use `json.dump()`.

---

## 📘 Code Example: Write to a JSON File

```python
import json

data = {
    "name": "Ali",
    "age": 21,
    "skills": ["Python", "AI", "Data"]
}

with open("output.json", "w") as file:
    json.dump(data, file, indent=4)
``` 

#### 📖 Explanation

 - `json.dump()` saves the dictionary to a file
 - `indent=4` makes the file human-readable
 - Very useful for saving configs, logs, or model metadata

---

## 🧪 Practice Exercises

### ✅ Exercise 1: Read a CSV and Print Formatted Output

**Goal:** Read rows and display them nicely.

```python
import csv

with open("students.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        name, age = row
        print(f"Name: {name}, Age: {age}")
```
**What you learn:** Extract and format CSV rows for display or processing.

---

### ✅ Exercise 2: Convert CSV to JSON

**Goal:** Read from a CSV and save to JSON format.

```python
import csv
import json

students = []

with open("students.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        students.append(row)

with open("students.json", "w") as json_file:
    json.dump(students, json_file, indent=4)
```

**What you learn:** Convert structured tabular data into nested format for APIs or ML.

---

### ✅ Exercise 3: Modify and Save JSON

**Goal:** Load JSON, update a value, and write back.

```python
import json

with open("settings.json", "r") as file:
    settings = json.load(file)

settings["theme"] = "dark"

with open("settings.json", "w") as file:
    json.dump(settings, file, indent=4)
``` 

**What you learn:** Real-world pattern of modifying config files or responses.

---

## 🔗 Related Files

🐍 [Day9_DataFiles/csv_json_examples.py](../Day9_DataFiles/Day9_DataFiles/csv_json_examples.py) - All examples in one script

---

## 🧠 Summary

| Concept        | Details                                |
| -------------- | -------------------------------------- |
| csv.reader     | Reads rows as lists                    |
| csv.writer     | Writes data row-by-row                 |
| csv.DictReader | Reads rows as dictionaries (key=value) |
| json.load()    | Reads JSON into Python objects         |
| json.dump()    | Saves Python objects into JSON format  |
| indent=4       | Formats JSON for better readability    |


✅ You can now work with structured data using CSV and JSON - a crucial skill for AI, APIs, and real-world datasets!

📅 **Next: Day 10 - Numpy for AI** - learn to work with powerful arrays and numerical data!