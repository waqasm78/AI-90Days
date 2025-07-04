# day7_exceptions.py

# Example 1: Basic try-except
try:
    x = int("abc")
except ValueError:
    print("Invalid input! Please enter a number.")

# Example 2: else and finally blocks
try:
    result = 10 / 2
except ZeroDivisionError:
    print("Cannot divide by zero.")
else:
    print("Result is:", result)
finally:
    print("This always runs.")

# Example 3: Multiple exceptions
try:
    x = int("abc")
    y = 10 / 0
except ValueError:
    print("That's not a number.")
except ZeroDivisionError:
    print("Cannot divide by zero.")

# Example 4: Raising your own exception
def set_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative.")
    print("Age set to:", age)

try:
    set_age(-1)
except ValueError as ve:
    print("Caught an error:", ve)
