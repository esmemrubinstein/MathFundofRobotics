import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

points = np.loadtxt('/Users/esmerubinstein/Desktop/MathFundofRobotics/hw3/gecko_robot_data/cluttered_hallway.txt')
remaining_points = points.copy()

num_iterations = 10000
distance_threshold = 0.01  # meters
num_planes = 4

planes = []          # plane coefficients (a,b,c,d)
inliers_list = []    # inlier points per plane
smoothness_scores = []

colors = ['blue', 'green', 'red', 'purple']

for plane_idx in range(num_planes):
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
        distances = np.abs(a*x + b*y + c*z + d) / np.linalg.norm(normal)
        inliers = np.where(distances < distance_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (a, b, c, d)

    if best_plane is None or len(best_inliers) == 0:
        print(f"No more planes found at iteration {plane_idx}.")
        break

    inlier_points = remaining_points[best_inliers]
    x_in, y_in, z_in = inlier_points[:,0], inlier_points[:,1], inlier_points[:,2]
    A = np.c_[x_in, y_in, np.ones_like(x_in)]
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    S_inv = np.diag(1 / S)
    A_pinv = VT.T @ S_inv @ U.T
    coeff = A_pinv @ z_in
    alpha, beta, gamma = coeff
    a_ref, b_ref, c_ref, d_ref = -alpha, -beta, 1.0, -gamma

    planes.append((a_ref, b_ref, c_ref, d_ref))
    inliers_list.append(inlier_points)

    residuals = np.abs(a_ref*x_in + b_ref*y_in + c_ref*z_in + d_ref) / np.sqrt(a_ref**2 + b_ref**2 + c_ref**2)
    smoothness = np.std(residuals)
    smoothness_scores.append(smoothness)

    print(f"Plane {plane_idx+1}: {a_ref:.4f}x + {b_ref:.4f}y + {c_ref:.4f}z + {d_ref:.4f} = 0")
    print(f"Inliers: {len(best_inliers)}, Smoothness: {smoothness:.6f} m")

    remaining_points = np.delete(remaining_points, best_inliers, axis=0)

safest_idx = np.argmin(smoothness_scores)
print(f"\nSafest surface: Plane {safest_idx+1} (lowest smoothness = {smoothness_scores[safest_idx]:.6f} m)")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

if len(remaining_points) > 0:
    ax.scatter(remaining_points[:,0], remaining_points[:,1], remaining_points[:,2],
               color='gray', alpha=0.3, s=2, label='Other points')

for idx, inlier_points in enumerate(inliers_list):
    ax.scatter(inlier_points[:,0], inlier_points[:,1], inlier_points[:,2],
               color=colors[idx % len(colors)], s=5, label=f'Plane {idx+1} inliers')

    a, b, c, d = planes[idx]
    xx, yy = np.meshgrid(np.linspace(inlier_points[:,0].min(), inlier_points[:,0].max(), 20),
                         np.linspace(inlier_points[:,1].min(), inlier_points[:,1].max(), 20))
    zz = (-a*xx - b*yy - d)/c
    ax.plot_surface(xx, yy, zz, color=colors[idx % len(colors)], alpha=0.4)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Four Detected Planes and Smoothness Scores')
ax.legend()
plt.show()
