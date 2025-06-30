# Day 6 – Modules and Packages in Python 📦

Welcome to **Day 6**! Today we will learn how to organize your code using **modules** and **packages**. This is an essential step in building real-world AI applications where your codebase becomes too large for a single file.

---

## 🌟 Objectives

- Understand what Python modules and packages are
- Learn how to create and import them
- Use built-in modules from the Python Standard Library
- Organize your AI project code into clean, reusable components
- Learn how to set up everything using Jupyter Notebook or Visual Studio 2022

---

## 🔍 What is a Module?

A **module** is simply a `.py` file (Python script) that contains variables, functions, or classes you want to use elsewhere. Instead of writing a huge script, you split your code into smaller modules and import them where needed.

This is very important in AI when your code has many parts, like data preprocessing, model training, visualization, etc.

### 🔧 Real-World Analogy

Think of a module like a **toolbox**. If you’re building something, you don’t carry every tool all the time — you grab the one you need. Likewise, you can import only the tools (functions or variables) that you need from a module.

---

## ✏️ How to Create a Module (Step-by-Step)

### ▶️ Using Visual Studio 2022:

1. Open your Visual Studio 2022
2. Create/Open a Python environment/project
3. Inside the `Day6_Modules` folder, right-click > Add > New Item > Python File
4. Name it `math_utils.py`

Add this code:

```python
def square(x):
    return x * x

def cube(x):
    return x * x * x
```

5. When you create a Python project, by default, Visual Studio creates a `Day6_Modules.py` file, Rename it to `main.py` and add the following code.

```python
import math_utils

print(math_utils.square(4))
print(math_utils.cube(3))
```

### ▶️ Using Jupyter Notebook:

1. Go to your `Day6_Modules` folder using terminal:
```bash
cd AI-90Days
ai_env\Scripts\activate
cd Day6_Modules
set JUPYTER_RUNTIME_DIR=%CD%\jupyter_runtime
jupyter notebook
```
2. From Jupyter Dashboard:

    - Click `New > Text File` → rename it to `math_utils.py`
    - Add the same `square` and `cube` functions
    - Click `New > Python 3 Notebook` → rename to `day6_modules.ipynb`
    - In the notebook:

```python
import math_utils
print(math_utils.square(5))
```

---

## 📦 What is a Package?

A **package** is a directory (folder) that contains multiple **modules** and a special file named `__init__.py`.

---

### 📁 Basic Package Example:

```
Day6_Modules/
│
├── my_package/
│   ├── __init__.py
│   └── greetings.py
└── main.py
```

## 🧩 What is `__init__.py` and Why Do We Need It?

In Python, packages are just directories that contain `.py` files (modules). However, to tell Python that a directory is a **package**, you must include a special file called `__init__.py`.

### 🎯 Purpose of `__init__.py`:

| Feature                         | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| 🏷️ **Marks the folder as a package**   | Tells Python to treat the folder as a proper package.                        |
| 🔁 **Controls imports**               | You can pre-import specific classes/functions to simplify access.            |
| ⚙️ **Runs initialization code**        | Code inside `__init__.py` runs when the package is imported.                 |
| 🧼 **Can be empty**                  | If you don’t need special behavior, you can leave it completely empty.       |

### 💡 Real-World Analogy:

If a module is a **toolbox**, a package is like a **tool shed** with multiple labeled toolboxes.

The `__init__.py` file is like the **main label or instruction sheet** in that shed—it lets Python know that everything inside belongs together.

### How to create a package (Step-by-Step):

#### In Visual Studio:

1. Right-click on `Day6_Modules` → Add → New Folder → Name it `my_package`
2. Inside `my_package`, add two new files:
    - `__init__.py` (leave empty)
    - `greetings.py` and the following code:
  
```python
def say_hello(name):
    return f"Hello, {name}!"
```

3. In `main.py`, use:

```python
from my_package import greetings
print(greetings.say_hello("Waqas"))
```

#### In Jupyter Notebook:

1. Use File > New > Folder and rename it to `my_package`
2. Create two new files:
    - `__init__.py` (empty)
    - `greetings.py` with the above function
3. Create a notebook or Python script and test the import

---

## 🧰 Using Built-in Modules

Python gives you many built-in modules so you don’t need to write everything yourself.

```python
import math
print(math.sqrt(25))

import random
print(random.randint(1, 10))

import os
print(os.getcwd())
```

These are fast, tested, and widely used.

---

## ✅ Full Setup Guide

1. Activate virtual environment:
```bash
cd AI-90Days
ai_env\Scripts\activate
mkdir Day6_Modules
cd Day6_Modules
set JUPYTER_RUNTIME_DIR=%CD%\jupyter_runtime
jupyter notebook
```

2. Create:

    - `math_utils.py` → for reusable math functions
    - `main.py` or `day6_modules.ipynb` → to test imports

3. Create package:

    - Folder `my_package`
    - Files: `__init__.py` and `greetings.py`

4. Test code with relative import

```python
from my_package import greetings
print(greetings.say_hello("Ali"))
```

---

## 🧪 Practice Exercises (with code and explanation)

---

### 📘 Exercise 1: Create a Utility Module – `string_utils.py`

**Goal:** Learn how to create a basic module and import it.

#### 🔧 Step-by-step:

1. Inside `Day6_Modules` folder, create a file called `string_utils.py`.

2. Add the following code:

```python
# string_utils.py

def reverse_string(s):
    """Returns the reverse of the input string."""
    return s[::-1]

def count_vowels(s):
    """Returns the number of vowels in a string."""
    return sum(1 for char in s.lower() if char in 'aeiou')
```

3. Now, create a new file called `test_utils.py` or use a new Jupyter notebook cell:

```python
import string_utils

print(string_utils.reverse_string("Hello AI"))      # Output: IA olleH
print(string_utils.count_vowels("Artificial"))      # Output: 5
```

#### 🧠 What you learned:

You just created a reusable utility module for string functions.
You imported it and used the functions from another file or notebook.

---

### 📘 Exercise 2: Create a Simple Package – ai_helpers

**Goal:** Understand how to create a package with multiple modules and use it.

#### 🔧 Step-by-step:

1. Create a folder called ai_helpers inside `Day6_Modules`.
2. Inside it, create an empty `__init__.py` file.
3. Create a new file: `math_ops.py`

```python
# ai_helpers/math_ops.py

def normalize(data):
    """Normalize a list of numbers to 0–1 range."""
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]
```

4. Create another file: `text_ops.py` in the `Day6_Modules` folder.

```python
# ai_helpers/text_ops.py

def clean_text(text):
    """Removes punctuation and converts to lowercase."""
    import string
    return text.translate(str.maketrans('', '', string.punctuation)).lower()

def word_count(text):
    """Counts words in a string."""
    return len(text.split())
```

5. Create a new file `test_ops.py` inside `Day6_Modules` folder and the following code.

```python
from ai_helpers import math_ops, text_ops

data = [5, 10, 15]
print("Normalized:", math_ops.normalize(data))  # [0.0, 0.5, 1.0]

sentence = "Hello, AI World!"
print("Cleaned:", text_ops.clean_text(sentence))  # hello ai world
print("Word Count:", text_ops.word_count(sentence))  # 3
```

#### 🧠 What you learned:

You built a real Python package ai_helpers with two modules and imported them properly using relative paths.

---

### 📘 Exercise 3: Use Built-in Module – datetime

#### 🔧 Step-by-step:

1. Open a new `test_datetime.py` file inside Day6_Modules.
2. Add the following code:

```python
import datetime

# Print the current date and time
now = datetime.datetime.now()
print("Current Timestamp:", now)

# Format the date nicely
formatted = now.strftime("%A, %d %B %Y, %I:%M %p")
print("Formatted:", formatted)
```

#### 🧠 What you learned:

Built-in Python modules like datetime save you time.
They come with many helpful functions ready to use without installation.

---

## 📘 Related Working Files

👉 Open [day6_modules.ipynb](../Day6_Modules/day6_modules.ipynb) to interactively test module/package creation.

---

## 🧠 Summary

- **Module** = single Python file
- **Package** = folder with modules and `__init__.py`
- Organize code into modules for maintainability
- Use packages to group related functionality
- Jupyter or Visual Studio both work — choose based on comfort

📅 On **Day 7**, we will explore **Object-Oriented Programming (OOP)** in Python!

