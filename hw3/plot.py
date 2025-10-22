import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x) - 0.5

x = np.linspace(-np.pi/2, np.pi/2, 500)
y = f(x)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, label=r"$f(x) = \sin x - 0.5$", color="blue")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")  # x-axis
plt.axvline(0, color="black", linewidth=0.8, linestyle="--")  # y-axis
plt.title(r"Graph of $f(x) = \sin x - 0.5$ over $[-\pi/2, \pi/2]$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
