import numpy as np

# --- Creating 1D and 2D Arrays ---
print("1D and 2D Arrays")
a = np.array([1, 2, 3, 4])
print("1D Array:", a)

b = np.array([[1, 2], [3, 4]])
print("2D Array:\n", b)

# --- Basic Array Operations ---
print("\nBasic Array Operations")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("Addition:", a + b)
print("Multiplication:", a * b)
print("Power:", a ** 2)

# --- Array Properties ---
print("\nArray Properties")
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("Shape:", arr.shape)
print("Size:", arr.size)
print("Dimensions:", arr.ndim)

# --- Indexing and Slicing ---
print("\nIndexing and Slicing")
arr = np.array([10, 20, 30, 40, 50])
print("Element at index 2:", arr[2])
print("Slice [1:4]:", arr[1:4])

# --- Reshaping Arrays ---
print("\nReshaping Arrays")
arr = np.array([1, 2, 3, 4, 5, 6])
reshaped = arr.reshape(2, 3)
print("Reshaped Array:\n", reshaped)

# --- NumPy Functions: Mean, Max, Min, Sum ---
print("\nStatistical Functions")
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("Mean:", np.mean(arr))
print("Max:", np.max(arr))
print("Min:", np.min(arr))
print("Sum:", np.sum(arr))

# --- Random Array ---
print("\nRandom Array")
rand_arr = np.random.rand(2, 3)
print("Random 2x3 Array:\n", rand_arr)

# --- Practice Exercise 1: Create and Multiply Arrays ---
print("\nPractice 1: Multiply Arrays")
a = np.array([2, 4, 6])
b = np.array([1, 2, 3])
result = a * b
print("Result:", result)

# --- Practice Exercise 2: Reshape and Slice ---
print("\nPractice 2: Reshape and Slice")
arr = np.arange(1, 10)
reshaped = arr.reshape(3, 3)
print("Array:", arr)
print("Matrix:\n", reshaped)
print("Middle Row:", reshaped[1])

# --- Practice Exercise 3: Random Matrix and Sum ---
print("\nPractice 3: Random Matrix and Sum")
rand_data = np.random.randint(1, 10, (3, 3))
print("Random Matrix:\n", rand_data)
print("Sum of all elements:", np.sum(rand_data))

