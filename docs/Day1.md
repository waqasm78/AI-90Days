# Day 1 – Environment Setup & First Jupyter Notebook 🚀

Welcome to Day 1 of your AI journey! Today’s goal is to:

* Install all required tools
* Set up your virtual environment
* Launch your first Jupyter Notebook using Visual Studio 2022 and command line

---

## 🎯 Objectives

* Install Python 3.11+
* Set up Visual Studio 2022 with Python workload
* Install Jupyter
* Set up a clean folder structure
* Launch your first notebook using the correct working directory

---

## 🛠 Tools & Environment Setup

### 1. Install Python 3.11+

Python is the most widely used language in AI/ML due to its simplicity and vast ecosystem.

🔗 Download: [https://www.python.org/downloads](https://www.python.org/downloads)

During installation:
✅ Check **“Add Python to PATH”** to make Python accessible from the terminal.

To confirm installation:

```bash
python --version
```

### 2. Install Visual Studio 2022 with Python Development Workload

Visual Studio 2022 is a powerful IDE ideal for developers transitioning from C#, C++, or .NET.

🔗 Download: [https://visualstudio.microsoft.com/vs/](https://visualstudio.microsoft.com/vs/)

During installation:

* Select **"Python development"** workload
* Optionally, select **"Data science and analytical applications"** for extra tools

**Why Visual Studio 2022?**

* Full IDE experience with IntelliSense, debugging, and project templates
* Jupyter notebook support inside the IDE
* Familiar environment for .NET/C++ developers

---

## ⚙️ Create Virtual Environment

A virtual environment isolates Python dependencies for your project, preventing conflicts with system-wide packages.

From the `AI-90Days/` folder:

```bash
python -m venv ai_env
```

To activate the environment (Windows):

```bash
ai_env\Scripts\activate
```

Once activated, your terminal prompt changes to indicate you're now inside `ai_env`:

```
(ai_env) C:\Users\...\AI-90Days>
```

Now any Python packages you install will remain local to this environment.

---

## 📦 Install Jupyter Notebook

Jupyter Notebooks allow you to write and run code in small steps with outputs and visualizations displayed inline.

While in the virtual environment:

```bash
pip install notebook
```

You can now launch Jupyter using the command below. But first…

---

## 🚫 Fixing Jupyter Permission Error (Important!)

Some systems show a `PermissionError` when Jupyter tries to write runtime files to its default location (`AppData`). To prevent this:

1. Create a runtime directory inside your project:

```bash
mkdir jupyter_runtime
```

2. Set an environment variable to force Jupyter to use this directory:

```bash
set JUPYTER_RUNTIME_DIR=%CD%\jupyter_runtime
```

This will ensure Jupyter’s server logs and HTML files don’t cause permission issues.

If this step is skipped, launching Jupyter may result in:

```plaintext
PermissionError: [Errno 13] Permission denied: 'C:\Users\<username>\AppData\Roaming\jupyter\runtime\...'
```

✅ **Always run Jupyter after setting this variable to avoid issues**.

---

## 📁 Set Up Folder Structure

To keep your work organized, follow this structure:

```plaintext
AI-90Days/
├── ai_env/                ← virtual environment (shared across days)
├── Day1_Setup/            ← work files for today
├── jupyter_runtime/       ← Jupyter runtime override directory
├── docs/                  ← contains Day1.md, Day2.md, etc.
```

From the `AI-90Days/` folder:

```bash
mkdir Day1_Setup
cd Day1_Setup
```

---

## 🚀 Launch Jupyter Notebook from the Correct Directory

Make sure your virtual environment is activated and `JUPYTER_RUNTIME_DIR` is set.

From inside the `Day1_Setup` folder:

```bash
jupyter notebook
```

This opens your browser at something like:

```
http://localhost:8888/tree
```

Here’s what to do:

* Click **New > Python 3 Notebook**
* Rename it to: `hello_day1.ipynb`
* In the first cell, type:

```python
print("Hello, AI World! This is Day 1.")
```

* Run the cell (Shift + Enter)
* Click `File > Save and Checkpoint`

Your notebook will now be saved inside `Day1_Setup`.

### 📂 What is `.ipynb_checkpoints`?

When you save a notebook, Jupyter creates a hidden folder named `.ipynb_checkpoints` in the same directory.

It contains backup versions of your notebooks (like `hello_day1-checkpoint.ipynb`) so that you can recover your work if something goes wrong.

📌 To ignore these from version control, add the following line to a `.gitignore` file in the root of your project:

```
.ipynb_checkpoints/
```

You can create this file using Notepad or any text editor.

---

## 🧠 Recap

Today you:

* Installed Python and Visual Studio 2022
* Created your virtual environment
* Installed and fixed Jupyter Notebook permissions
* Launched your first notebook from the correct folder
* Created and saved your first `.ipynb` file 🎉

Tomorrow: Python Basics — variables, data types, and simple operations.

---

🔁 **Next Time Tip:**
Each day, follow this pattern:

```bash
cd AI-90Days
ai_env\Scripts\activate
set JUPYTER_RUNTIME_DIR=%CD%\jupyter_runtime
cd Day2_PythonBasics
jupyter notebook
```

Happy Learning! 💡
