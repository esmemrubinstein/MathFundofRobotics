import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

def load_paths(filename):
    """
    returns a list of paths, each as a dict with keys:
        'x' : np.ndarray of shape (50,)
        'y' : np.ndarray of shape (50,)
    """
    paths = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for i in range(0, len(lines), 2):
        x = np.fromstring(lines[i], sep=' ')
        y = np.fromstring(lines[i+1], sep=' ')
        paths.append({'x': x, 'y': y})
    return paths


def interpolate_path_linear(path, t):
    """
    Simple linear interpolation for a path at continuous time t.
    path: dict with keys 'x', 'y' (length 50)
    t: scalar or array of times
    Returns: np.array([x_t, y_t])
    """
    t_grid = np.arange(len(path['x']))
    x_t = np.interp(t, t_grid, path['x'])
    y_t = np.interp(t, t_grid, path['y'])
    return np.array([x_t, y_t])



def classify_path_side(path, center=(5.0, 5.0), radius=1.5, tol=1e-6):
    """
    Decide if a path goes 'left' or 'right' of the ring center,
    using only points whose y lies in [center_y - radius, center_y + radius].

    Returns 'left' or 'right'.
    Fallback: if no in-band points, uses global mean x.
    """
    x = np.asarray(path['x'])
    y = np.asarray(path['y'])
    cx, cy = center

    # define band
    y_min = cy - radius - tol
    y_max = cy + radius + tol
    in_band = (y >= y_min) & (y <= y_max)

    if np.any(in_band):
        x_in = x[in_band]
        dx = x_in - cx

        pos_count = np.sum(dx > tol)  
        neg_count = np.sum(dx < -tol)   

        if pos_count > neg_count:
            return 'right'
        elif neg_count > pos_count:
            return 'left'
        else:
            return 'invalid_path'
    else:
        # No samples in the vertical band, fall back to global mean x but this is not great option
        global_mean_x = np.mean(x)
        return 'left' if global_mean_x < cx else 'right'



def point_in_triangle(point, tri):
    """
    Compute coordinates (alpha, beta, gamma) of 'point'
    w.r.t. triangle defined by tri = [(x1,y1),(x2,y2),(x3,y3)].
    Returns (alpha, beta, gamma) or None if triangle is degenerate.
    """
    x, y = point
    (x1, y1), (x2, y2), (x3, y3) = tri
    A = np.array([
        [x1, x2, x3],
        [y1, y2, y3],
        [1,  1,  1 ]
    ])
    b = np.array([x, y, 1])
    try:
        v = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return v


def triangle_area(tri):
    """
    Compute area of triangle given by three 2D points.
    For choosing which triangle when multiple options
    """
    (x1, y1), (x2, y2), (x3, y3) = tri
    return 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))


def find_valid_triangle(start, starts, sides):
    """
    Search all triangles of starting points to find one containing 'start'.
    Returns (i, j, k, bary) for the smallest-area valid triangle.
    """
    candidate_tris = []
    for (i, j, k) in combinations(range(len(starts)), 3):
        if not (sides[i] == sides[j] == sides[k]):
            continue
        tri = [starts[i], starts[j], starts[k]]
        bary = point_in_triangle(start, tri)
        if bary is None:
            continue
        if np.all(bary >= -1e-10):  # inside or on edge
            area = triangle_area(tri)
            candidate_tris.append((area, i, j, k, bary))
    if not candidate_tris:
        raise ValueError("No valid triangle containing the start position found.")
    candidate_tris.sort(key=lambda x: x[0])
    _, i, j, k, bary = candidate_tris[0]
    return i, j, k, bary


def weighted_path_at_t(t, paths, indices, weights):
    """
    Compute weighted sum of 3 paths at time t.
    indices: tuple (i, j, k) of paths
    weights: np.array([alpha_i, alpha_j, alpha_k])
    """
    i, j, k = indices
    alpha_i, alpha_j, alpha_k = weights
    pi = interpolate_path_linear(paths[i], t)
    pj = interpolate_path_linear(paths[j], t)
    pk = interpolate_path_linear(paths[k], t)
    return alpha_i * pi + alpha_j * pj + alpha_k * pk


ring_center = (5, 5)
ring_radius = 1.5

paths = load_paths("/Users/esmerubinstein/Desktop/MathFundofRobotics/hw2/paths.txt")
starts = [(p['x'][0], p['y'][0]) for p in paths]
sides = [classify_path_side(p, center=(5,5), radius=1.5) for p in paths]

test_starts = [(0.8, 1.8), (2.2, 1.0), (2.7, 1.4)]

# ---- Define fixed axis limits ----
x_limits = (0, 12)
y_limits = (0, 12)

for start in test_starts:
    # Find valid triangle + bary weights
    i, j, k, bary = find_valid_triangle(start, starts, sides)

    # Generate interpolated path
    t_vals = np.linspace(0, 49, 200)  # finer resolution
    interp_path = np.array([weighted_path_at_t(t, paths, (i, j, k), bary) for t in t_vals])

    # Plot
    fig, ax = plt.subplots()
    
    # Ring of fire
    circle = plt.Circle(ring_center, ring_radius, color='r', fill=False, linewidth=2)
    ax.add_artist(circle)

    # Original paths used
    for idx, color in zip((i, j, k), ['b', 'g', 'm']):
        ax.plot(paths[idx]['x'], paths[idx]['y'], color=color, linestyle='--', label=f"path {idx}")

    # Interpolated path
    ax.plot(interp_path[:, 0], interp_path[:, 1], 'k', linewidth=2, label="interpolated")

    # Start point
    ax.scatter(*start, c='k', marker='x', s=100, label="start")

    # Fixed axis limits
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_aspect('equal')

    ax.legend()
    ax.set_title(f"Interpolated path for start {start}")
    plt.show()
