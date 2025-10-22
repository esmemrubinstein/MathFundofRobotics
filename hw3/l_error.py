import numpy as np

a = 0.0
b = 24 / np.pi**3
c = -0.5

def p_ls(x):
    return a * x**2 + b * x + c

def f(x):
    return np.sin(x) - 0.5

# Interval
x_min, x_max = -np.pi/2, np.pi/2
x_vals = np.linspace(x_min, x_max, 10000)
errors = f(x_vals) - p_ls(x_vals)

L_inf = np.max(np.abs(errors))
# trap integration
L2 = np.sqrt(np.trapz(errors**2, x_vals))

print(f"L-infinity error: {L_inf:.6f}")
print(f"L2 error: {L2:.6f}")
