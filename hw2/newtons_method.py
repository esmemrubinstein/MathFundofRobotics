import numpy as np

def f(x):
    return x - np.tan(x)

def f_prime(x):
    return 1 - (1/np.cos(x))**2


def newtons_method(x0, tol=1e-10, max_iter=100):
    x = x0
    for i in range(max_iter):
        y = f(x)
        y_prime = f_prime(x)
        if y_prime == 0:
            raise ValueError("Derivative is zero. No solution found.")
        x_new = x - y / y_prime
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    
    return x  

low_guess = 14
high_guess = 17.27

low = newtons_method(low_guess)
high = newtons_method(high_guess)

print(f"Low root: {low}")
print(f"High root: {high}")
