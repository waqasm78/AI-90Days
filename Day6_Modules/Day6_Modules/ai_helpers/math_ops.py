# ai_helpers/math_ops.py

def normalize(data):
    """Normalize a list of numbers to 0-1 range."""
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]
