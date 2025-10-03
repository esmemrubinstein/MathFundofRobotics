import numpy as np

def muller(func, x0, x1, x2, tol=1e-10, max_iter=100):
    # cast everything to complex
    x0, x1, x2 = complex(x0), complex(x1), complex(x2)

    for _ in range(max_iter):
        h1 = x1 - x0
        h2 = x2 - x1 
        delta1 = (func(x1) - func(x0)) / h1
        delta2 = (func(x2) - func(x1)) / h2

        d = (delta2 - delta1) / (h2 + h1)

        b = delta2 + h2 * d
        D = np.lib.scimath.sqrt(b**2 - 4 * func(x2) * d)  # allow complex sqrt

        if abs(b - D) < abs(b + D):
            E = b + D
        else:
            E = b - D

        h = -2 * func(x2) / E
        x3 = x2 + h

        if abs(h) < tol:
            return x3
        
        x0, x1, x2 = x1, x2, x3

    raise ValueError("Muller's method did not converge")


def poly_func(coeffs):
    def f(x):
        return np.polyval(coeffs, x)
    return f


def find_all_roots(coeffs, init_guesses=(0, 0.5, 1), tol=1e-10):
    """
    coeffs: list of polynomial coefficients (highest degree first)
    init_guesses: tuple of 3 starting guesses for Müller's method
    tol: tolerance for convergence
    """
    roots = []
    coeffs = np.array(coeffs, dtype=complex)  # allow complex coefficients
    coeffs_original = coeffs.copy()
    n = len(coeffs) - 1  # degree

    while n > 0:
        f = poly_func(coeffs)
        r = muller(f, *init_guesses, tol=tol)

        # deflate
        coeffs, remainder = np.polydiv(coeffs, [1, -r])
        if abs(remainder).max() > 1e-6:
            print(f"Warning: large remainder {remainder}")

        roots.append(r)
        n -= 1

    f_orig = poly_func(coeffs_original)
    polished = []
    for r in roots:
        polished_r = muller(f_orig, r, r + 1e-3, r - 1e-3, tol=tol)
        polished.append(polished_r)

    return polished


coeffs = [1, 0, 1, 1]

roots = find_all_roots(coeffs, init_guesses=(-1, -0.5, 0))
print("Roots:", roots)
