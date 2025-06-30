import datetime

# Print the current date and time
now = datetime.datetime.now()
print("Current Timestamp:", now)

# Format the date nicely
formatted = now.strftime("%A, %d %B %Y, %I:%M %p")
print("Formatted:", formatted)