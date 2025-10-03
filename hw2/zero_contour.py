import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 3, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)

p = 2*X**2 + 2*Y**2 - 4*X - 4*Y + 3
q = X**2 + Y**2 + 2*X*Y - 5*X - 3*Y + 4

# Plot contours
contour_p = plt.contour(X, Y, p, levels=[0], colors='blue')
contour_q = plt.contour(X, Y, q, levels=[0], colors='red')

# Add legend using proxy lines
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='blue', lw=2, label='p=0'),
                   Line2D([0], [0], color='red', lw=2, label='q=0')]
plt.legend(handles=legend_elements)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Zero contours of p(x,y) and q(x,y)')
plt.grid()
plt.show()
