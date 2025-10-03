import sympy as sp

x, y = sp.symbols('x y')
p = 2*x**2 + 2*y**2 - 4*x - 4*y + 3
q = x**2 + y**2 + 2*x*y - 5*x - 3*y + 4
p_coeffs = sp.Poly(p, y).all_coeffs()  
q_coeffs = sp.Poly(q, y).all_coeffs()

a2, a1, a0 = p_coeffs
b2, b1, b0 = q_coeffs
S = sp.Matrix([
    [a2, a1, a0, 0],
    [0, a2, a1, a0],
    [b2, b1, b0, 0],
    [0, b2, b1, b0]
])

res = S.det()
sp.pprint(res)

x_solutions = sp.solve(res, x)
# sp.pprint(x_solutions)
x_numeric = [xi.evalf() for xi in x_solutions]
print("Numeric x-coordinates of intersections:")
print(x_numeric)

intersection_points = []
for xi in x_numeric:
    y_sols = sp.solve(p.subs(x, xi), y)
    for yi in y_sols:
        # Only keep points that satisfy q(x,y)=0
        if abs(q.subs({x: xi, y: yi}).evalf()) < 1e-12:
            intersection_points.append((xi, yi.evalf()))

print("Numeric intersection points (x, y):")
print(intersection_points)


solutions = sp.solve([p, q], (x, y))
print("Intersection points (symbolic):")
print(solutions)

numeric_solutions = [(s[0].evalf(), s[1].evalf()) for s in solutions]
print("Intersection points (numeric):")
print(numeric_solutions)