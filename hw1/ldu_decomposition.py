import numpy as np


def ldu_decomposition(A):

    A = A.astype(float)
    n = A.shape[0]
    L = np.eye(n)
    P = np.eye(n)
    Aprime = A.copy()

    for k in range(n):
        print(Aprime, "step", k)
        if Aprime[k, k] == 0:
            # swap rows when pivot is zero
            for i in range(k+1, n):
                if Aprime[i, k] != 0:
                    Aprime[[k, i], :] = Aprime[[i, k], :]
                    if k > 0:
                        L[[k, i], :k] = L[[i, k], :k]
                    P[[k, i], :] = P[[i, k], :]
                    break
        if Aprime[k, k] == 0:
            raise ValueError("Matrix cannot be decomposed.")
        pivot = Aprime[k, k]
        for i in range(k+1, n):
            multiplier = Aprime[i, k] / pivot
            L[i, k] = multiplier
            Aprime[i, :] = Aprime[i, :] - multiplier * Aprime[k, :]

    D = np.diag(np.diag(Aprime))

    U = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            U[i, j] = Aprime[i, j] / D[i, i]

    print("P =\n", P)
    print("L =\n", L)
    print("D =\n", D)
    print("U =\n", U)
    print("\n")
    return P, L, D, U


# Examples
A = np.array([[1, 1, 0],
              [1, 1, 2],
              [4, 2, 3]])

P, L, D, U = ldu_decomposition(A)
# verify
print("A reconstructed =\n", P.T @ L @ D @ U)  # should equal to A

A_1 = np.array([[10, -10, 0],
                [0, -4, 2],
                [2, 0, -5]])

P, L, D, U = ldu_decomposition(A_1)
print("A_1 reconstructed =\n", P.T @ L @ D @ U)  # should equal to A_1
