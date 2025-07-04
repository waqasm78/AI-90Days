# custom_errors.py

class InsufficientFundsError(Exception):
    """Raised when the balance is not enough for withdrawal"""
    pass

def withdraw(balance, amount):
    if amount > balance:
        raise InsufficientFundsError("Not enough balance!")
    return balance - amount

# Test the custom error
try:
    withdraw(500, 700)
except InsufficientFundsError as e:
    print("Transaction failed:", e)
