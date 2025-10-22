import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


points = np.loadtxt('/Users/esmerubinstein/Desktop/MathFundofRobotics/hw3/gecko_robot_data/clean_hallway.txt')
x_all, y_all, z_all = points[:,0], points[:,1], points[:,2]


num_iterations = 1000
distance_threshold = 0.01  # meters
num_planes = 4


remaining_points = points.copy()
print(points.shape)
planes = []          # list of plane coefficients (a,b,c,d)
plane_inliers = []   # list of inlier indices for each plane
colors = ['blue', 'green', 'red', 'purple']


for i in range(num_planes):
    best_inliers = []
    best_plane = None
    x, y, z = remaining_points[:,0], remaining_points[:,1], remaining_points[:,2]

    for _ in range(num_iterations):
        sample_idx = np.random.choice(len(remaining_points), 3, replace=False)
        p1, p2, p3 = remaining_points[sample_idx]

        normal = np.cross(p2 - p1, p3 - p1)
        if np.linalg.norm(normal) < 1e-6:
            continue
        a, b, c = normal
        d = -np.dot(normal, p1)

        # distances to plane
        distances = np.abs((a*x + b*y + c*z + d)) / np.linalg.norm(normal)
        inliers = np.where(distances < distance_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (a, b, c, d)

    if best_plane is None or len(best_inliers) == 0:
        print(f"Plane {i+1} not found, stopping early.")
        break

    inlier_points = remaining_points[best_inliers]
    x_in, y_in, z_in = inlier_points[:,0], inlier_points[:,1], inlier_points[:,2]
    A = np.c_[x_in, y_in, np.ones_like(x_in)]
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    S_inv = np.diag(1 / S)
    A_pinv = VT.T @ S_inv @ U.T
    coeff = A_pinv @ z_in

    alpha, beta, gamma = coeff
    a, b, c, d = -alpha, -beta, 1.0, -gamma

    planes.append((a, b, c, d))
    plane_inliers.append(inlier_points)

    # remove inliers from remaining points
    remaining_points = np.delete(remaining_points, best_inliers, axis=0)
    print(f"Plane {i+1}: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0, inliers = {len(best_inliers)}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

if len(remaining_points) > 0:
    ax.scatter(remaining_points[:,0], remaining_points[:,1], remaining_points[:,2],
               color='gray', alpha=0.4, s=2, label='Other points')

for i, inlier_points in enumerate(plane_inliers):
    ax.scatter(inlier_points[:,0], inlier_points[:,1], inlier_points[:,2],
               color=colors[i % len(colors)], s=5, label=f'Plane {i+1} inliers')

    a, b, c, d = planes[i]
    xx, yy = np.meshgrid(np.linspace(inlier_points[:,0].min(), inlier_points[:,0].max(), 20),
                         np.linspace(inlier_points[:,1].min(), inlier_points[:,1].max(), 20))
    zz = (-a * xx - b * yy - d) / c
    ax.plot_surface(xx, yy, zz, color=colors[i % len(colors)], alpha=0.4)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Four Dominant Planes in Hallway')
ax.legend()
plt.show()
