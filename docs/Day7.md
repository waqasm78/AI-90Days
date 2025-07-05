# 📅 Day 7 – Exception Handling in Python

---

## 🧠 What You’ll Learn

Today, you'll learn how to gracefully handle errors in your Python programs using a mechanism called **Exception Handling**.

You’ll understand:

- What exceptions are and why we need to handle them
- How to use `try`, `except`, `else`, and `finally`
- How to raise your own exceptions with `raise`
- How to create custom exception classes
- Practice exercises to reinforce concepts

---

## ✅ Why We Need Exception Handling

Every program has a chance of failing at runtime due to things like:

- Invalid user input
- Missing files
- Network issues
- Dividing by zero

Without exception handling, such failures **crash your program**. With it, you can show **friendly messages** and safely continue or stop.

**💡 Analogy:** Think of exception handling like air bags in a car — they won't prevent an accident, but they reduce the damage and help you recover.

---

## 🛠 Setup in Visual Studio 2022

1. Open **Visual Studio 2022**
2. Create a new **Python Application**
3. Name it: `Day7_Exceptions`
4. Save in: `AI-90Days\Day7_Exceptions`
5. Create two files manually in Solution Explorer:
   - `day7_exceptions.py`
   - `custom_errors.py`

Optional: Also create a Jupyter Notebook if using Jupyter: `day7_exceptions.ipynb`

---

## 🔍 What Is an Exception?

An **exception** is an error that occurs **while the program is running**.

Example:

```python
num = 10
result = num / 0   # ❌ ZeroDivisionError
```

This error will stop your entire program unless you catch and handle it. Python uses special keywords to help you handle exceptions. These are `try`, `except`, `else`, and `finally`. Let's break them down.

---

## 🧪 try-except Block

The `try` block is where you put the code that might cause an exception. It's like saying, "I'm going to try to do this, but I know it might go wrong." If an exception occurs inside the `try` block, Python immediately jumps to the `except` block. This is where you put the code to handle the specific problem.

### ✅ Basic Structure

```python
try:
    # code that may fail
except ErrorType:
    # what to do if it fails
```

Python tries to run the code inside `try`. If any error occurs, it jumps to the `except` block.

### 🧑‍💻 Example:

```python
try:
    x = int("abc")
except ValueError:
    print("Invalid input! Please enter a number.")
```

🧠 **Explanation:** Trying to convert a string `"abc"` to an integer causes `ValueError`. We catch it and show a user-friendly message instead of crashing.

---

## ➕ else and finally Blocks

The `else` block is optional. The code inside the `else` block will only run if no exception occurred in the `try` block. The `finally` block is also optional. The code inside the `finally` block will always run, regardless of whether an exception occurred or not, and even if an exception was handled. This is perfect for cleanup operations, like closing files or releasing resources.

You can also use `else` and `finally` with `try-except`:

```python
try:
    result = 10 / 2
except ZeroDivisionError:
    print("Cannot divide by zero.")
else:
    print("Result is:", result)
finally:
    print("This always runs.")
```

🔍 Explanation:

- **`try`**: Runs the risky code.
- **`except`**: Handles errors.
- **`else`**: Runs if no error occurred.
- **`finally`**: Always runs (for cleanup like closing files or disconnecting a network).

---

## 🔄 Handling Multiple Exceptions

You can have multiple `except` blocks to handle different types of exceptions:

```python
try:
    x = int("abc")
    y = 10 / 0
except ValueError:
    print("That's not a number.")
except ZeroDivisionError:
    print("Cannot divide by zero.")
```

🧠 **Explanation:** If multiple errors might occur, handle them in separate blocks. Python stops at the first error it encounters.

You can also group them:

```python
except (ValueError, TypeError):
    print("A value or type error occurred.")
```

---

## 🚨 Raising Your Own Exceptions

You can use `raise` to create errors when something is logically wrong.

```python
def set_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative.")
    print("Age set to:", age)
```

🧠 **Explanation:** If someone tries to set a negative age, we **raise** an error intentionally. This helps keep data valid.

---

## 🛠 Custom Exceptions

You can define your own error types using classes:

```python
class InsufficientFundsError(Exception):
    pass

def withdraw(balance, amount):
    if amount > balance:
        raise InsufficientFundsError("Not enough balance!")
    return balance - amount
```

🧠 **Explanation:** Instead of using built-in errors, define a meaningful name for your error. It makes your code easier to understand.

---

## 📝 Practice Exercises

### 1. Handle File Not Found

```python
try:
    f = open("data.txt")
    print(f.read())
except FileNotFoundError:
    print("File not found.")
```

📌 If the file doesn’t exist, this prevents your app from crashing.

---

### 2. Validate User Input

```python
while True:
    try:
        age = int(input("Enter your age: "))
        break
    except ValueError:
        print("That’s not a valid number. Try again.")
```

📌 Keeps asking until the user provides a valid number — a great way to make your app user-friendly.

---

### 3. Custom Exception for Bank App

```python
class InsufficientBalance(Exception):
    pass

def withdraw(balance, amount):
    if amount > balance:
        raise InsufficientBalance("Insufficient balance.")
    return balance - amount

try:
    withdraw(1000, 1500)
except InsufficientBalance as e:
    print("Transaction Failed:", e)
```

📌 Demonstrates using your own error class in a real-world example like an ATM.

---

## 📘 Summary

- Use `try` and `except` to catch and handle errors.
- Add `else` for success code and `finally` for cleanup.
- Raise your own exceptions with `raise`.
- Create custom error types for meaningful messages.
- Practice writing robust code with validations.

---

## 📂 Project Files

```
AI-90Days/
└── Day7_Exceptions/
    ├── day7_exceptions.py
    └── custom_errors.py
```

### 🔗 Related Files

- [Day7_Exceptions/day7_exceptions.py](../Day7_Exceptions/day7_exceptions.py) – Code examples for Visual Studio 2022  
- [Day7_Exceptions/custom_errors.py](../Day7_Exceptions/custom_errors.py) – Custom error class example