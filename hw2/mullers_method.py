import numpy as np

def f(x):
    return x**3 + x + 1

def muller(x0, x1, x2, tol=1e-10, max_iter=100):
    for _ in range(max_iter):
        h1 = x1 - x0
        h2 = x2 - x1 
        delta1 = (f(x1) - f(x0)) / h1
        delta2 = (f(x2) - f(x1)) / h2

        d = (delta2 - delta1) / (h2 + h1)

        b = delta2 + h2 * d
        D = np.sqrt(b**2 - 4 * f(x2) * d)

        if abs(b - D) < abs(b + D):
            E = b + D
        else:
            E = b - D

        h = -2 * f(x2) / E
        x3 = x2 + h

        if abs(h) < tol:
            return x3
        
        x0, x1, x2 = x1, x2, x3

    raise ValueError("Muller's method did not converge")


r1 = muller(-1, -0.5, 0)
print(f"Root 1: {r1}")
