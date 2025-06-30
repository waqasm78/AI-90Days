# ai_helpers/text_ops.py

def clean_text(text):
    """Removes punctuation and converts to lowercase."""
    import string
    return text.translate(str.maketrans('', '', string.punctuation)).lower()

def word_count(text):
    """Counts words in a string."""
    return len(text.split())
