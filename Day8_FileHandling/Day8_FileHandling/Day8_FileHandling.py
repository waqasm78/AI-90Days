# 1. Reading a File

file = open("sample.txt", "r")
print(file.read())
file.close()

# 2. Reading a File using with statement

with open("sample.txt", "r") as file:
    content = file.read()
    print(content)

# 3 Reading a file line by line using a loop

with open("sample.txt", "r") as file:
    for line in file:
        print(line.strip())

# 4 Reading a file line by line using a readline

with open("sample.txt", "r") as file:
    lines = file.readlines()
    print(lines)

# 5 Writing to a file

with open("output.txt", "w") as file:
    file.write("Hello, AI World!\n")
    file.write("This is Day 8.\n")

# 6 Appending to a File

from datetime import datetime
with open("output.txt", "a") as file:
    file.write(f"\nDateTime: {datetime.now()}")

# 7 Reading a Structured Text

with open("students.txt", "r") as file:
    for line in file:
        name, age = line.strip().split(",")
        print(f"Name: {name}, Age: {age}")

# 8 Practice Exercise 1: Write User Input to File

with open("names.txt", "w") as file:
    for i in range(3):
        name = input("Enter a name: ")
        file.write(name + "\n")

# 9 Practice Exercise 2:

with open("names.txt", "r") as file:
    lines = file.readlines()
    print("Total lines:", len(lines)) 

# 10 Practice Exercise 3: 

with open("data.txt", "r") as file:
    text = file.read()

text = text.replace("Python", "AI")

with open("data_updated.txt", "w") as file:
    file.write(text)