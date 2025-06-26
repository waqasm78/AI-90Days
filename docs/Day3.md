# Day 3 – Python Collections: Lists, Tuples, Sets & Dictionaries 📦

Welcome to Day 3 of your AI journey! Today is all about **Python collections**, which are containers used to store multiple values in one variable. In AI and Data Science, we deal with datasets, features, and structured inputs – so mastering these collections is essential.

---

## 🎯 Objectives

* Understand the four core built-in collection types in Python:

  * `list`, `tuple`, `set`, and `dict`
* Learn how to create, update, access, and iterate through these collections
* Apply them to real-life use cases
* Practice with exercises and an interactive notebook
* Learn how to identify which type is being used based on Python syntax

---

## ⚙️ Python vs. .NET: Key Difference in Collection Syntax

In C#/.NET, collections are declared using **explicit types** like:

```csharp
List<string> myList = new List<string>();
Dictionary<string, int> scores = new Dictionary<string, int>();
```

But in **Python**, collections are identified by the **symbols** used:

| Collection Type | Syntax Example     | Key Symbols          |
| --------------- | ------------------ | -------------------- |
| List            | `[1, 2, 3]`        | Square brackets `[]` |
| Tuple           | `(1, 2, 3)`        | Round brackets `()`  |
| Set             | `{1, 2, 3}`        | Curly braces `{}`\*  |
| Dictionary      | `{"key": "value"}` | Key-value in `{}`    |

> ⚠️ Tip: To avoid confusion between sets and dictionaries, remember:
>
> * Sets have **only values** inside `{}` (no colons)
> * Dictionaries have **key: value pairs** inside `{}`

> 📌 Note: An **empty set** is created with `set()` — `{}` creates an **empty dictionary** by default.

---

## 📁 Setup Instructions

Before starting:

1. Open **Command Prompt** or **Anaconda Prompt** and activate your virtual environment:

```bash
cd AI-90Days
ai_env\Scripts\activate
```

> ✅ This ensures all your packages and configurations stay isolated to this AI learning project.

2. Create a new folder for today’s work:

```bash
mkdir Day3_Collections
cd Day3_Collections
```

> 📁 This folder will hold your notebook and any scripts for Day 3.

3. Fix Jupyter permission issue (if encountered):

```bash
set JUPYTER_RUNTIME_DIR=%CD%\jupyter_runtime
```

> 📌 This step helps prevent PermissionError when launching Jupyter. It explicitly sets the runtime directory to your current folder.

4. Start Jupyter Notebook:

```bash
jupyter notebook
```

> If you see a browser permission error like:

```
PermissionError: [Errno 13] Permission denied
```

✅ **Solution:**

* Run Command Prompt as **Administrator**
* Activate the environment from inside the working directory
* Use `set JUPYTER_RUNTIME_DIR=%CD%\jupyter_runtime` to avoid permission conflicts

5. The browser will open the Jupyter dashboard (if not, copy the link shown in terminal with `http://localhost:8888/...`).

6. In the Jupyter dashboard:

   * Click **New > Python 3 (ipykernel)** in the top right.
   * Rename the notebook to `day3_collections.ipynb` by clicking the title and choosing **Rename**.

✅ The notebook will automatically be saved in the `Day3_Collections` folder.

---

## 🧰 What is a List?

A **list** is an ordered, changeable (mutable) collection that can store multiple values.

### ✅ When to use:

* You want to store multiple values in one place
* You need to change (add/remove/update) those values later

### 🔤 Example:

```python
fruits = ["apple", "banana", "cherry"]  # create a list
print(fruits[0])  # Output: apple (access by index)
fruits.append("orange")  # add an item
print(fruits)     # ['apple', 'banana', 'cherry', 'orange']
```

### 🔁 Loop through a list

```python
for fruit in fruits:
    print(fruit)
```

This prints each item one by one.

### 🛠️ Useful List Methods

```python
fruits.remove("banana")   # removes an item
print(len(fruits))        # number of items
fruits.sort()             # sorts the list alphabetically
```

---

## 🔐 What is a Tuple?

A **tuple** is like a list, but it is **immutable** – it cannot be changed after creation.

### ✅ When to use:

* Your data should never change (e.g., coordinates)
* You want better performance (tuples are faster)

### 🔤 Example:

```python
coordinates = (10.0, 20.5)  # create a tuple
print(coordinates[1])       # Output: 20.5
```

### ⚠️ Try this (will give error):

```python
coordinates[0] = 11.0  # ❌ Error: tuple is immutable
```

---

## 🌱 What is a Set?

A **set** is an unordered collection of **unique** items.

### ✅ When to use:

* You want to eliminate duplicates
* You want to do set math like union, intersection, difference

### 🔤 Example:

```python
colors = {"red", "green", "blue", "green"}  # duplicate 'green' ignored
print(colors)  # Output: {'red', 'green', 'blue'}
colors.add("yellow")
```

### 🔁 Set Operations

```python
odd = {1, 3, 5}
even = {2, 4, 6}
print(odd.union(even))        # combine both sets
print(odd.intersection(even)) # common elements (none)
```

---

## 🗃️ What is a Dictionary?

A **dictionary** stores data in `key: value` pairs.

### ✅ When to use:

* You want to label your data (e.g., name → value)
* You need fast access using keys

### 🔤 Example:

```python
student = {"name": "Ali", "age": 20, "grade": "A"}
print(student["name"])    # Output: Ali
student["age"] = 21        # update value
```

### 🔁 Loop through dictionary

```python
for key, value in student.items():
    print(key, ":", value)
```

---

## 🧪 Practice Exercises

### ✅ Exercise 1: List of Squares

```python
squares = []
for i in range(1, 11):
    squares.append(i**2)
print(squares)
```

This creates a list of squares from 1² to 10².

### ✅ Exercise 2: Dictionary Lookup

```python
person = {"name": "Sara", "job": "Engineer"}
key = input("Enter key: ")
print(person.get(key, "Key not found"))
```

`get()` safely looks for a key and avoids errors if not found.

### ✅ Exercise 3: Common Items with Sets

```python
a = [1, 2, 3, 4]
b = [3, 4, 5, 6]
common = set(a).intersection(b)
print(list(common))
```

This converts both lists to sets and finds their intersection.

---

## 📓 Related Notebook

👉 [Open day3\_collections.ipynb](../Day3_Collections/day3_collections.ipynb) to see all examples with code and try them live in Jupyter.

---

## 🧠 Summary

* **Lists** – ordered, changeable → use for dynamic data (created using square brackets `[]`)
* **Tuples** – ordered, unchangeable → use for fixed records (created using parentheses `()`)
* **Sets** – unordered, unique → use to eliminate duplicates or compare groups (created using curly braces `{}` without colons)
* **Dictionaries** – key-value pairs → use for labeled data (created using curly braces `{}` **with** colons `:`)

You now understand the backbone of data handling in Python. Tomorrow we’ll explore **control flow** with `if`, `for`, `while`, and logical expressions.

Keep going strong! 💪
