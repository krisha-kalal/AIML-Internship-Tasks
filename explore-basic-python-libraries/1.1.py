import numpy as np 
arr = np.array([1,2,3])
print(arr)

# arange: Creates an array with a range of values.
print("arange():", np.arange(1, 10, 2))

# concatenate: Combines two arrays.
a = np.array([4,5,6])
print("concatenate():", np.concatenate((arr, a)))

# ones: Creates an array of all the elements as 1.
print("ones():", np.ones(4))

# empty: Creates an uninitialized array.
print("empty():", np.empty((4, 3)))

# put: Replaces elements in the array at specified indices.
arr = np.array([1, 2, 3]) 
np.put(arr, [2], [-9]) 
print("put():", arr)

# min & max: Finds the minimum and maximum element from an array.
print("min():", np.min(arr))
print("max():", np.max(arr))

# reshape: Reshapes the array to the specified dimensions.
arr1 = np.array([(1, 5, 6), (8, 2, 4)]) 
b = arr1.reshape(3, 2) 
print("reshape(): ", b)

# reverse: Reverses the elements of an array.
print("reverse:", arr[::-1])

# delete: Deletes an element from array at a specified index.
print("deleting element:", np.delete(a, 1))

# isalnum: Checks if each string contains only alphanumeric characters.
str_arr = np.array(['Hello123', 'World456']) 
print("char.isalnum():", np.char.isalnum(str_arr))

# strip: Removes all the leading and trailing whitespaces in a given string.
str_arr = np.array(['Hello       ', '        World']) 
print("char.strip():", np.char.strip(str_arr))

# replace: Replaces old string with new string.
print("char.replace():", np.char.replace(str_arr, "o", "0"))
