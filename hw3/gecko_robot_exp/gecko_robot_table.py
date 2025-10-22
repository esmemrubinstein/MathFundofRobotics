import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

points = np.loadtxt('/Users/esmerubinstein/Desktop/MathFundofRobotics/hw3/gecko_robot_data/cluttered_table.txt')
x, y, z = points[:, 0], points[:, 1], points[:, 2]
A = np.c_[x, y, np.ones_like(x)]

# svd
U, S, VT = np.linalg.svd(A, full_matrices=False)
S_inv = np.diag(1 / S)
A_pinv = VT.T @ S_inv @ U.T

coeff = A_pinv @ (-z)  
a, b, d = coeff

print(f"Plane equation (implicit form): {a:.6f}x + {b:.6f}y + 1.000000z + {d:.6f} = 0")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, color='b', s=2, label='Data points')

xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 20),
                     np.linspace(y.min(), y.max(), 20))
zz = -a * xx - b * yy - d  # since we set c=1

ax.plot_surface(xx, yy, zz, alpha=0.5, color='orange', label='Fitted plane')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
