import numpy as np
from scipy.optimize import fsolve

def equation(x):
    return np.sin(x) - x * np.cos(x) - (np.pi / 2) * np.cos(x) + 1

# initial guess (somewhere between 0 and pi/2)
x0 = 0.7

x_star = fsolve(equation, x0)[0]

b = np.cos(x_star)
c = -0.5
E = (np.pi / 2) * b - 1

print(f"x* = {x_star:.9f}")
print(f"b  = {b:.9f}")
print(f"c  = {c:.9f}")
print(f"E  = {E:.9f}")
