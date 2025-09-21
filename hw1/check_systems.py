import numpy as np

x = np.array([0.018, 0.860, 0.123])
A = np.array([[1, 1, 0],
              [1, 1, 2],
              [4, 2, 3]])

print("Ax = ", A @ x)

A = np.array([[10, -10, 0],
              [0, -4, 2],
              [2, 0, -5]])

A_inverse = la.inv(A)
print("A_inverse =\n", A_inverse)
b = np.array([-10, 8, -7])