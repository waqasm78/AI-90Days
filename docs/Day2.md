# Day 2 – Python Basics: Variables, Data Types & Operations 🐍

Welcome to Day 2 of your AI learning journey! Today, we will build the foundational skills of Python by understanding variables, data types, and basic operations.

---

## 🎯 Objectives

* Understand what variables are and how to define them
* Learn Python’s core data types: `int`, `float`, `str`, `bool`
* Perform basic arithmetic and string operations
* Learn type checking and type casting with explanations
* Learn how to create and manage Jupyter notebooks properly
* Learn how to execute code cells inside a Jupyter Notebook

---

## 📁 Setup Reminder

Before continuing, make sure you:

1. Open **Command Prompt** or **Visual Studio Developer Command Prompt**
2. Navigate to your AI project folder:

   ```bash
   cd AI-90Days
   ai_env\Scripts\activate
   set JUPYTER_RUNTIME_DIR=%CD%\jupyter_runtime
   mkdir Day2_PythonBasics
   cd Day2_PythonBasics
   jupyter notebook
   ```
3. Your browser will open and show the Jupyter dashboard (at `http://localhost:8888/tree`)
4. On the top-right of the dashboard, click `New > Python 3 (ipykernel)` to create a new notebook
5. Click on `Untitled` at the top of the new notebook tab, rename it to `day2_basics.ipynb`, and click **Rename**
6. This file will be saved in your `Day2_PythonBasics` folder by default

If the notebook fails to open, make sure you are not running as Administrator and that Jupyter has permission to write to the `runtime` folder. You can set it explicitly like we did with `set JUPYTER_RUNTIME_DIR`.

---

## ▶️ How to Execute Code in Jupyter Notebook

Once you're inside the notebook interface:

1. **Click on a code cell** (you’ll see `In [ ]:` on the left).
2. **Write your Python code** (e.g. `print("Hello AI World!")`).
3. **Run the cell** using one of these methods:

   * Press `Shift + Enter` → runs the current cell and moves to the next
   * Press `Ctrl + Enter` → runs the current cell but stays there
   * Click the **Run ▶️ button** in the top toolbar

👉 After execution, the output will appear directly under the cell. The `In [ ]:` turns into something like `In [1]:` to show it's been run.

This is how you will run all code examples and practice exercises below.

---

## 🧠 What is a Variable?

A variable in Python is a **named container** that holds a value. You can change its value anytime, and Python automatically detects the type.

Example:

```python
x = 5
```

Explanation:

* `x` is a variable
* `5` is an integer value assigned to `x`
* Python automatically understands that `x` is of type `int`

You don’t need to define the type explicitly — Python is dynamically typed.

---

## 🗂️ Python Data Types

Python supports different data types for different kinds of information.

### 1. Integers and Floats

```python
age = 30       # integer (int)
height = 5.9   # float (decimal number)
```

Explanation:

* `age` is a whole number, so it’s an `int`
* `height` includes a decimal, so it’s a `float`

### 2. Strings

```python
name = "Alice"
```

Explanation:

* `name` stores a sequence of characters
* Anything inside double or single quotes is treated as a string

### 3. Booleans

```python
is_student = True
is_graduated = False
```

Explanation:

* `True` and `False` are Boolean values used for logic
* They're often used in conditions like `if`, `while`, etc.

---

## 🔍 Type Checking

Want to confirm the type of a variable? Use `type()`:

```python
print(type(name))    # Output: <class 'str'>
print(type(height))  # Output: <class 'float'>
```

This tells you what type Python has assigned to each variable.

---

## 🔄 Type Conversion (Casting)

You can convert one data type into another. This is known as type casting.

### Example 1: String to Integer

```python
x = "100"          # x is a string
y = int(x)          # y becomes an integer 100
print(type(y))      # <class 'int'>
```

Explanation:

* `int(x)` tries to interpret the string as a number
* Useful when reading numbers from user input, which is always a string

### Example 2: Integer to String

```python
num = 25            # num is an integer
text = str(num)     # text becomes "25"
print(type(text))   # <class 'str'>
```

Explanation:

* `str()` is used when you want to join numbers into text or print with other strings

---

## ➕ Arithmetic Operations

Python supports common arithmetic operators:

```python
x = 10
y = 3
print(x + y)    # 13 (Addition)
print(x - y)    # 7  (Subtraction)
print(x * y)    # 30 (Multiplication)
print(x / y)    # 3.333... (Division - always returns float)
print(x // y)   # 3 (Floor Division - rounds down)
print(x % y)    # 1 (Modulus - remainder of division)
print(x ** y)   # 1000 (Exponentiation - 10 to the power 3)
```

Explanation:

* `+`, `-`, `*`, `/` are basic math
* `//` is useful when you want whole numbers only (e.g., index positions)
* `%` is helpful in finding even/odd or repeating patterns
* `**` is used in scientific computations

---

## 🧪 Practice Exercise 1: Declare and Inspect

Try this in your notebook:

```python
my_name = "Waqas"
my_age = 38
is_learning_ai = True

print("Name:", my_name)
print("Age:", my_age)
print("Learning AI:", is_learning_ai)

print(type(my_name))
print(type(my_age))
print(type(is_learning_ai))
```

---

## 🧪 Practice Exercise 2: Simple Calculator

Build a mini calculator that shows all operations:

```python
a = 8
b = 4

print("Sum:", a + b)
print("Difference:", a - b)
print("Product:", a * b)
print("Quotient:", a / b)
print("Floor Division:", a // b)
print("Remainder:", a % b)
print("Power:", a ** b)
```

Explanation:

* Try changing `a` and `b` to different values
* Observe the output and note any type changes (e.g., float vs int)

---

## 📓 Related Notebook

You can view and explore all examples and exercises interactively in this notebook:
👉 **[Open day2\_basics.ipynb](../Day2_PythonBasics/day2_basics.ipynb)**

---

## 🧠 Summary

Today you learned:

* How to declare and assign variables
* Common data types in Python
* How to check data types with `type()`
* How to convert between data types
* How to perform arithmetic calculations
* How to create and name your Jupyter notebooks in the correct folder
* How to run code cells in Jupyter Notebook using Shift+Enter or the Run button

Tomorrow: We’ll explore **collections** — lists, tuples, sets, and dictionaries — which help group and organize data.

You're doing great. Keep coding! 🚀
