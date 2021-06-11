#%%
import numpy as np

A = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
b = np.array([1, 2, 3])

print((A.T*b).T)
