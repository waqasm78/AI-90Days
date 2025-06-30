# string_utils.py

def reverse_string(s):
    """Returns the reverse of the input string."""
    return s[::-1]

def count_vowels(s):
    """Returns the number of vowels in a string."""
    return sum(1 for char in s.lower() if char in 'aeiou')