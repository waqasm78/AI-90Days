# Day 8 – File Handling in Python 📁

Welcome to **Day 8**! Today we will explore working with **text files** in Python. This includes reading from files, writing to files, and modifying files—all of which are foundational skills for real-world AI projects, like loading data, logging results, or saving outputs.

---

## 🌟 Objectives

- Understand how to open and close files
- Learn to read entire files or line by line
- Practice writing and appending data to files
- Learn to handle files safely using `with` statement
- Use Visual Studio 2022 or Jupyter Notebook to test file I/O

---

## 🔍 Why Learn File Handling?

In AI and data tasks, you'll often work with:

- Training datasets in `.txt`, `.csv`, `.json` formats  
- Log and output files  
- Configuration files  

Mastering file I/O is essential for loading data, saving predictions, and recording model information.

---

## ✏️ How to Read and Write Files (Step-by-Step)

### ▶️ Using Visual Studio 2022

1. Open Visual Studio 2022  
2. Create a new Python project called `Day8_FileHandling`  
3. Add these files:
   - `file_handling_basics.py`
   - `sample.txt`: add some sample text
   - `students.txt`: add lines like:
     ```
     Ali,21
     Sara,22
     John,20
     ```

---

## ✅ The Basic Steps of File Handling

Working with files in Python generally involves these three steps:

  1. **Opening the file:** You tell Python which file you want to work with and what you want to do with it (read, write, or both).
  2. **Performing operations (Read/Write):** Once the file is open, you can read content from it or write new content to it.
  3. **Closing the file:** This is a very important step! It saves changes and frees up the file so other programs can use it. Forgetting to close files can lead to data corruption or loss.

---

## 📄 Opening a File

To work with a file, you first need to open it using Python's built-in `open()` function. Python lets you open files in different modes depending on what you want to do.

 - `"r"`  = read
 - `"w"`  = write (overwrites if file exists and creates if it doesn't exist.)
 - `"a"`  = append (adds to end of file)
 - `"r+"` = read/write


```python
file = open("sample.txt", "r")
print(file.read())
file.close()
```

 - `"sample.txt"`: the file name
 - `"r"`: the mode (read mode)
 - `.read()`: reads the entire file content
 - `.close()`: closes the file to free resources

---

## 🔤 Reading Text Files

When you want to work with existing data, logs, or inputs stored in a `.txt` file, Python makes it very easy to **read** from these files.

### ⭐ Read Entire File

The `read()` reads the entire content of the file as a single string.  

```python
with open("sample.txt", "r") as file:
    content = file.read()
    print(content)
```

  - `"r"` mode opens file for reading
  - `with open(...) as file:` handles closing automatically. Reduces chances of memory leaks or forgetting to close a file

### 🧠 Why it matters:

Reads the full content at once. Good for small files and quick checks.

---

### ⭐ Read Line by Line

You can read files line by line using a loop.

```python
with open("sample.txt", "r") as file:
    for line in file:
        print(line.strip())
```

  - `for line in file:` iterates over each line
  - `.strip()` removes newline characters
  - Useful for processing data line by line

You can convert file content into a list of lines using `.readlines()`.

```python
with open("sample.txt", "r") as file:
    lines = file.readlines()
    print(lines)
```

 - Returns a list of strings (one per line)
 - Useful for batch processing lines of data
---

## ✍️ Writing to a File

### ⭐ Overwrite Content

The `write()` call adds content to the file. The `\n` adds a new line between entries.

```python
with open("output.txt", "w") as file:
    file.write("Hello, AI World!\n")
    file.write("This is Day 8.\n")
```
  - `"w"` mode writes new content and replaces any existing file

---

### ⭐ Append to a File

Use `"a"` mode to append data to the end without overwriting existing content.

```python
with open("output.txt", "a") as file:
    file.write("Another line added.\n")
```

---

## 🔄 Working with Structured Text (CSV-Like)

Assume `students.txt` contains:

```
Ali,21
Sara,22
John,20
```

To read and parse the data:

```python
with open("students.txt", "r") as file:
    for line in file:
        name, age = line.strip().split(",")
        print(f"Name: {name}, Age: {age}")
```

This pattern appears often in data-driven applications.

---

## 🧪 Practice Exercises

---

### ✅ Exercise 1: Write User Input to File

**Goal:** Get user input and save it to a file.

```python
with open("names.txt", "w") as file:
    for i in range(3):
        name = input("Enter a name: ")
        file.write(name + "\n")
```

What you learn: Save input to a file for future use.

---

### ✅ Exercise 2: Count Lines

**Goal:** Get all the lines from a file and print the line count.

```python
with open("names.txt", "r") as file:
    lines = file.readlines()
    print("Total lines:", len(lines))
```

**What you learn:** Use `readlines()` to load all lines, then count them.

---

### ✅ Exercise 3: Replace Words and Save

**Goal:** Read all the lines from a file, replace a word and then save the updated content to a new file.

```python
with open("data.txt", "r") as file:
    text = file.read()

text = text.replace("Python", "AI")

with open("data_updated.txt", "w") as file:
    file.write(text)
```

**What you learn:** Read-modify-write pattern in file handling.

---

## 🔗 Related Files

🐍 [Day8_FileHandling.py](../Day8_FileHandling/Day8_FileHandling/Day8_FileHandling.py) – All examples in one script


## 🧠 Summary

| Concept                   | Details                                       |
| ------------------------- | --------------------------------------------- |
| `with open(...):`         | Safely opens and closes files                 |
| `"r", "w", "a"` modes     | Read, Write (overwrite), Append               |
| `.read()`, `.readlines()` | Read full text or split lines                 |
| `.write()`                | Save new text to the file                     |
| `.strip()`                | Removes newline and spaces from ends of lines |


✅ You now know how to handle everyday file tasks in Python — a critical skill for AI and scripting.

📅 Next: **Day 9 - Working with CSV and JSON Files**, a step deeper into real-world data handling!