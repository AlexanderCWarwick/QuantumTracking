import numpy as np
import matplotlib.pyplot as plt


print("hello world")

def myfunc(a,b):
    return float(a + b)

class Error(Exception):
    def __init__(self, message):
        self.message = message
    

def new_func(c, d):
    if not isinstance(c, np.ndarray) or not isinstance(d, np.ndarray):
        raise Error("New_func numpy array arguements do not exist.")
    else:
        return c * d
    
print(new_func(np.array([2,3,4]), np.array([2,3,4])))
