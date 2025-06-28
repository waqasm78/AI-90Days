# Day 5 – Functions in Python 🚀

Welcome to **Day 5** of your AI journey! Today is all about **functions**. Functions are reusable blocks of code that help you organize and modularize your programs. In AI projects, functions are essential for keeping your logic clean, separating responsibilities, and reducing repetition.

---

## 🌟 Objectives

* Learn how to define and call Python functions
* Understand how to pass arguments and return values
* Explore advanced topics like default values, keyword arguments, and variable-length arguments
* Build real-world AI-relevant utilities using functions

---

## 📁 Setup Instructions

1. Activate your virtual environment and create a Day 5 folder:

```bash
cd AI-90Days
ai_env\Scripts\activate
mkdir Day5_Functions
cd Day5_Functions
set JUPYTER_RUNTIME_DIR=%CD%\jupyter_runtime
jupyter notebook
```

2. Create a new notebook: `day5_functions.ipynb`

---

## 🔍 What is a Function?

Functions are containers for code that perform specific tasks. Once defined, a function can be called multiple times, allowing code reuse and better structure.

This is especially useful in AI to organize tasks like loading data, preprocessing it, or evaluating models. Instead of rewriting the same code, you just call the function where needed.

---

## ✏️ Defining a Simple Function

Use the `def` keyword to define a function. Then call the function using its name.

```python
def greet():
    print("Hello from a function!")

greet()
```

This simple function prints a greeting message. Functions without parameters are useful when the logic is fixed.

---

## 🧰 Parameters and Arguments

Functions can take parameters (placeholders) so you can pass data when calling them.

```python
def greet_user(name):
    print("Hello,", name)

greet_user("Waqas")
```

Here, `name` is a parameter. You pass "Waqas" as an argument during the function call. This allows for dynamic input.

---

## ➡️ Returning Values

You can use the `return` keyword to send back a result from a function.

```python
def square(x):
    return x * x
```

Returning values is key when building pipelines in AI — you pass data from one function to another for transformation.

---

## 📢 Types of Arguments

Python offers several ways to pass arguments:

### 1. Default Arguments

Allow you to specify default values for parameters so the caller may skip them.

```python
def greet(name="Guest"):
    print("Hi", name)
```

### 2. Keyword Arguments

Use the parameter name explicitly, which improves readability.

```python
profile(age=30, name="Ali")
```

### 3. Variable-Length Arguments

Use `*args` to accept a variable number of arguments.

```python
def sum_all(*numbers):
    total = sum(numbers)
    print("Total:", total)
```

These features make functions highly flexible, especially in situations like passing hyperparameters in AI models.

---

## 🌐 Real-World Analogy

Think of a function like a coffee machine:

* You press a button (input)
* It processes beans, water, etc. (logic)
* You get your coffee (output)

Functions help abstract logic so that you don't need to know internal details — just use it!

---

## 🚀 AI Use Case Example: Normalization Function

Before feeding data to a model, we usually normalize it.

```python
def normalize(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]
```

This function scales values to a 0–1 range, a common preprocessing step in ML and neural networks.

---

## 🔹 Practice Exercises

### 1. Prime Checker

Create a function to check whether a number is prime.

```python
def is_prime(n):
    # logic to check prime
```

### 2. Factorial Function

Write a function that returns the factorial of a number using a loop.

### 3. Vowel Counter

Write a function that counts how many vowels are in a given string.

These are great mini-problems to sharpen your function skills.

---

## 📓 Related Notebook

👉 [Open day5\_functions.ipynb](../Day5_Functions/day5_functions.ipynb) to explore, run, and modify all the examples shown above interactively.

---

## 🧠 Summary

* Functions are essential tools to break down tasks and organize code
* They make large AI projects manageable and readable
* You can pass different types of arguments to customize behavior
* Return values allow you to create pipelines and workflows

📅 On Day 6, we will explore **modules and packages** to scale our codebases further!
