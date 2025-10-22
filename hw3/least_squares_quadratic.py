import numpy as np

pi = np.pi

A = np.array([
    [pi, 0, pi**3 / 12],
    [0, pi**3 / 12, 0],
    [pi**3 / 12, 0, pi**5 / 80]
])

d = np.array([
    -pi / 2,
    2,
    -pi**3 / 24
])

x = np.linalg.solve(A, d)

c, b, a = x 

print(f"a = {a:.6f}")
print(f"b = {b:.6f}")
print(f"c = {c:.6f}")