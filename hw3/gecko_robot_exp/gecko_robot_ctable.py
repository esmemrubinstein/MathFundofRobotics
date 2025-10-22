import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

points = np.loadtxt('/Users/esmerubinstein/Desktop/MathFundofRobotics/hw3/gecko_robot_data/cluttered_table.txt')
x, y, z = points[:, 0], points[:, 1], points[:, 2]

num_iterations = 1000
distance_threshold = 0.01  
best_inliers = []
best_plane = None


for _ in range(num_iterations):
    sample_indices = np.random.choice(len(points), 3, replace=False)
    p1, p2, p3 = points[sample_indices]

    normal = np.cross(p2 - p1, p3 - p1)
    if np.linalg.norm(normal) < 1e-6:
        continue  # skip collinear points

    a, b, c = normal
    d = -np.dot(normal, p1)

    distances = np.abs((a * x + b * y + c * z + d)) / np.linalg.norm(normal)
    inliers = np.where(distances < distance_threshold)[0]

    if len(inliers) > len(best_inliers):
        best_inliers = inliers
        best_plane = (a, b, c, d)

inlier_points = points[best_inliers]
x_in, y_in, z_in = inlier_points[:, 0], inlier_points[:, 1], inlier_points[:, 2]
A = np.c_[x_in, y_in, np.ones_like(x_in)]
U, S, VT = np.linalg.svd(A, full_matrices=False)
S_inv = np.diag(1 / S)
A_pinv = VT.T @ S_inv @ U.T
coeff = A_pinv @ z_in

alpha, beta, gamma = coeff
a, b, c, d = -alpha, -beta, 1.0, -gamma

print(f"Dominant plane (table): {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
print(f"Inliers detected: {len(best_inliers)} / {len(points)}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# all points (gray)
ax.scatter(x, y, z, color='gray', alpha=0.4, s=2, label='All points')

# table inliers (blue)
ax.scatter(x_in, y_in, z_in, color='blue', s=5, label='Table inliers')

# fitted plane (orange)
xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 20),
                     np.linspace(y.min(), y.max(), 20))
zz = -a * xx - b * yy - d
ax.plot_surface(xx, yy, zz, color='orange', alpha=0.5)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Dominant Plane Fit via RANSAC')
ax.legend()

plt.show()
