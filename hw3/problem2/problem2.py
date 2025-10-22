import numpy as np
import matplotlib.pyplot as plt

# Load the data
f_vals = np.loadtxt("/Users/esmerubinstein/Desktop/MathFundofRobotics/hw3/problem2.txt") 
x = np.linspace(0, 1, 101) 

A = np.column_stack([
    np.ones_like(x),  # constant
    x,                # linear
    np.sin(5*np.pi*x) # main oscillation
])

U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.diag(1 / S)
c = Vt.T @ S_inv @ U.T @ f_vals
print("Coefficients (SVD):", c)

f_approx = A @ c

plt.plot(x, f_vals, label='Original f(x)')
plt.plot(x, f_approx, '--', label='Approximation')
plt.legend()
plt.show()