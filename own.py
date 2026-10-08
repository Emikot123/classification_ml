import numpy as np
import pandas as pd

y_test = np.array([1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0])
y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1])

def accuracy_score(y_test, y_pred):
    bool_arr = y_test == y_pred
    array = bool_arr[bool_arr == True]
    return len(array) / len(bool_arr)

def confusion_matrix(y_test, y_pred):
    print(array)


confusion_matrix(y_test, y_pred)