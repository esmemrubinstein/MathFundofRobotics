import scipy.linalg as la
import numpy as np

def compute_svd(A):
    U, s, VT = la.svd(A)
    print("U =\n", U)
    print("s =\n", s)
    print("VT =\n", VT)

A = np.array([[5, -5, 0, 0], 
              [5, 5, 5, 0], 
              [0, -1, 4, 1],
              [0, 4, -1, 2],
              [0, 0, 2, 1]])
# compute_svd(A)


A = np.array([[1, 1, 1],
              [10, 2, 9],
              [8, 0, 7]])
# compute_svd(A)
U, s, VT = la.svd(A)
sigma = np.zeros(A.shape)
sigma[:len(s), :len(s)] = np.diag(s)
Sigma_inv = la.pinv(sigma)
A_inv = VT.T @ Sigma_inv @ U.T
b = np.array([3,2,2])
x = A_inv @ b
print("x =\n", x)



