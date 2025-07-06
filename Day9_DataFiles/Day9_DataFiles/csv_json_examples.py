import csv
import json

# 1. --- Reading CSV ---
print("Reading students.csv:")
with open("students.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

# 2. --- Writing CSV ---
print("\nWriting to output.csv:")
with open("output.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Age"])
    writer.writerow(["Ali", 21])
    writer.writerow(["Sara", 22])

# 3. --- Reading JSON ---
print("\nReading data.json:")
with open("data.json", "r") as file:
    data = json.load(file)
    print(data)

# 4. --- Writing JSON ---
print("\nWriting to output.json:")
data = {
    "name": "Ali",
    "age": 21,
    "skills": ["Python", "AI", "Data"]
}
with open("output.json", "w") as file:
    json.dump(data, file, indent=4)

# 5. --- CSV to JSON ---
print("\nConverting CSV to JSON:")
students = []
with open("students.csv", "r") as file:
    reader = csv.DictReader(file, fieldnames=["Name", "Age"])
    for row in reader:
        students.append(row)
with open("students.json", "w") as json_file:
    json.dump(students, json_file, indent=4)

# 6. --- Modify JSON ---
print("\nModifying settings.json:")
with open("settings.json", "r") as file:
    settings = json.load(file)

settings["theme"] = "dark"

with open("settings.json", "w") as file:
    json.dump(settings, file, indent=4)

# 7. Practice Exercise 1: Read a CSV and Print Formatted Output

with open("students.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        name, age = row
        print(f"Name: {name}, Age: {age}")

# 8. Practice Exercise 2: Convert CSV to JSON

students = []

with open("students.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        students.append(row)

with open("students.json", "w") as json_file:
    json.dump(students, json_file, indent=4)

# 9. Practice Exercise 3: Modify and Save JSON

with open("settings.json", "r") as file:
    settings = json.load(file)

settings["theme"] = "dark"

with open("settings.json", "w") as file:
    json.dump(settings, file, indent=4) 