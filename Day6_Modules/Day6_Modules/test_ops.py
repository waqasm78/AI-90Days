from ai_helpers import math_ops, text_ops

data = [5,10,15]
print("Normalized:", math_ops.normalize(data))

sentence = "Hello, AI World!"
print("Cleaned:", text_ops.clean_text(sentence))
print("Word Count:", text_ops.word_count(sentence))