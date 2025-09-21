import numpy as np


def divided_differences(x_set, y_set):
    n = len(x_set)
    coefs = y_set.copy()

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coefs[i] = (coefs[i] - coefs[i - 1]) / (x_set[i] - x_set[i - j])
    
    # print("Divided differences coefficients:", coefs)
    return coefs

def interpolate(x, x_set, coefs):
    n = len(x_set)
    result = coefs[-1]
    for i in range(n - 2, -1, -1):
        result = result * (x - x_set[i]) + coefs[i]
    
    # print(result)
    return result


def f(x):
    return 2/(1 + 9*x**2)


def estimate_error(n, error_points=10000):
    x_points = np.linspace(-1, 1, n+1)
    y_points = [f(x) for x in x_points]

    coefs = divided_differences(x_points, y_points)

    X = np.linspace(-1, 1, error_points)
    Y = [f(x) for x in X]
    p = [interpolate(x, x_points, coefs) for x in X]
    errors = np.abs(np.array(Y) - np.array(p))
    max_error = np.max(errors)
    # print(f"Max error with n={n}: {max_error}")
    return max_error



xs = [0, 1/8, 1/4, 3/8, 1/2]
ys = [np.cos(np.pi*x) for x in xs]   # cosine data
# coefs = divided_differences(xs, ys)
# interpolate(3/10, xs, coefs)
#print(np.cos(np.pi*3/10))  # true value


n = 40
xs = [(i*(2/n)-1) for i in range(n+1)]
ys = [f(x) for x in xs]
# coefs = divided_differences(xs, ys)
# interpolate(0.7, xs, coefs)
# print(f(0.7))  # true value

for n in [2,4,6,8,10,12,14,16,18,20,40]:
    print(f"n={n}, E_n ≈ {estimate_error(n):.6f}")