# Day 4 – Python Control Flow: If, For, While, and Logic 🔄

Welcome to **Day 4** of your AI learning path! Today we will dive into the concept of control flow in Python, which determines the order in which code executes. Control flow is essential in any program to handle decisions, iterate over data, and repeat operations. It's particularly important in AI and data science for tasks like filtering data, making predictions based on conditions, and running training loops.

---

## 🌟 Objectives

* Understand **if-else conditions** for decision-making
* Learn **for loops** and **while loops** for repeating actions
* Explore **logical operators** for combining multiple conditions
* Practice with real-world examples and AI-relevant problems

---

## 📁 Setup Instructions

Before we begin coding, make sure to:

1. **Activate Environment & Create Folder**
   Open Command Prompt and run:

```bash
cd AI-90Days
ai_env\Scripts\activate
mkdir Day4_ControlFlow
cd Day4_ControlFlow
set JUPYTER_RUNTIME_DIR=%CD%\jupyter_runtime
jupyter notebook
```

2. **In Jupyter Notebook**

* Create a new file called `day4_control_flow.ipynb`
* Save it in the `Day4_ControlFlow` folder
* Use this notebook to follow and test all the examples below

---

## 🔐 Understanding Indentation in Python

Python uses **indentation** (spacing) instead of curly braces `{}` to mark code blocks. This makes code more readable but also means you must be careful:

```python
if x > 0:
    print("Positive")  # indented = inside the if block
```

> Indentation must be consistent (4 spaces recommended). Incorrect indentation will cause syntax errors.

---

## ✅ If-Else Statements

**Conditional statements** help the program make decisions based on certain conditions.

You use:

* `if` to test a condition
* `elif` (else if) for additional conditions
* `else` if all conditions fail

### Example:

```python
x = 10
if x > 0:
    print("Positive")
elif x == 0:
    print("Zero")
else:
    print("Negative")
```

This structure is foundational in AI for decision-making logic, such as choosing an action based on prediction scores.

---

## 📒 Comparison and Logical Operators

To create conditions, we use **comparison operators** (to compare values) and **logical operators** (to combine conditions).

### Comparison Operators:

| Operator | Meaning          |
| -------- | ---------------- |
| `==`     | Equal            |
| `!=`     | Not equal        |
| `>`      | Greater than     |
| `<`      | Less than        |
| `>=`     | Greater or equal |
| `<=`     | Less or equal    |

### Logical Operators:

| Operator | Description                         |
| -------- | ----------------------------------- |
| `and`    | Both conditions must be true        |
| `or`     | At least one condition must be true |
| `not`    | Reverses the condition              |

### Example:

```python
age = 25
has_id = True
if age > 18 and has_id:
    print("Allowed to enter")
```

Logical operators help when building AI logic — for instance, combining multiple feature checks in classification.

---

## ⟳ For Loop

A **for loop** lets you iterate over sequences like lists, tuples, strings, or ranges.

This is useful when you need to repeat an action for each item in a dataset.

### Example:

```python
for i in range(5):
    print("Iteration:", i)
```

This loop prints numbers from 0 to 4.

### Looping through a list:

```python
colors = ["red", "blue", "green"]
for color in colors:
    print("Color:", color)
```

You’ll use loops in AI to iterate through rows of data or epochs of training.

---

## ⏲️ While Loop

A **while loop** repeats as long as a condition is true. It's useful when the number of repetitions isn't known in advance.

### Example:

```python
count = 0
while count < 5:
    print("Count is", count)
    count += 1
```

While loops are useful for retries, waiting for data, or looping until convergence in AI algorithms.

---

## 🚧 Break and Continue

These are control tools to manage loops:

* `break` exits the loop early
* `continue` skips to the next iteration

### Example:

```python
for i in range(10):
    if i == 5:
        break   # stops loop
    if i % 2 == 0:
        continue  # skips even numbers
    print(i)
```

These tools are useful when scanning for certain values or handling edge cases in datasets.

---

## 🔧 Real-World AI Example: Data Filtering

You can filter data based on conditions — very common in AI for cleaning datasets or filtering results.

### Example:

```python
scores = [50, 85, 90, 30, 60]
passed = []
for score in scores:
    if score >= 60:
        passed.append(score)
print("Passed Scores:", passed)
```

Here we select only scores >= 60, which mimics filtering valid data points.

---

## 🔹 Practice Exercises

Try these to build your coding muscle:

```python
# Exercise 1: Even or Odd
num = int(input("Enter a number: "))
if num % 2 == 0:
    print("Even")
else:
    print("Odd")

# Exercise 2: Multiples of 5
for i in range(1, 11):
    print("5 x", i, "=", i * 5)

# Exercise 3: Password Retry
password = "python123"
attempt = ""
while attempt != password:
    attempt = input("Enter password: ")
print("Access granted!")
```

These simple exercises build a strong foundation for logic-based AI operations.

---

## 📓 Related Notebook

👉 [Open day4\_control\_flow.ipynb](../Day4_ControlFlow/day4_control_flow.ipynb) to view all examples and test them interactively.

---

## 🧠 Summary

* Use `if`, `elif`, `else` to add decision-making to your code
* Loop using `for` and `while` to repeat logic efficiently
* Apply logical operators (`and`, `or`, `not`) to combine multiple conditions
* Use `break` and `continue` to fine-tune your loop flow
* These structures are the base of nearly every AI/ML algorithm

✅ Tomorrow we move on to **functions** — a critical step in writing reusable, modular AI code.
