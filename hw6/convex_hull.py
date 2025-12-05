# 2D convex hull
# Input: finite set of 2D points
# Output: polygon constituting the boundary of the minimal convex set containing all the input points
import numpy as np
import matplotlib.pyplot as plt


# utility functions
def cross_product(o, a, b):
    """
    - Positive if counter-clockwise turn,
    - Negative if clockwise turn,
    - Zero if O, A, and B are collinear.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def polar_angle(origin, point):
    """
    polar angle of point with respect to origin.
    angle in radians from the positive x-axis.
    """
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    return np.arctan2(dy, dx)


def distance_squared(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def distance(p1, p2):
    return np.sqrt(distance_squared(p1, p2))


def polar_sort_key(point, anchor):
    """
    key for a point based on polar angle and distance from anchor.
    """
    if point == anchor:
        return -np.pi, 0 
    angle = polar_angle(anchor, point)
    dist = distance_squared(anchor, point)
    return angle, dist


def graham_scan(points):
    """
    Inputs: list of tuples [(x1, y1), (x2, y2), ...] or array of points
    Gives: list of points that form the convex hull in counter-clockwise order
    """
    # check for at least 3 points for a convex hull
    if len(points) < 3:
        return list(points)
    
    points = [tuple(p) for p in points]
    
    # find the starting point (lowest y-coordinate, leftmost if tie)
    anchor = min(points, key=lambda p: (p[1], p[0]))
    
    # sort points by polar angle with respect to anchor
    # break ties by distance (closer points first)
    sorted_points = sorted(points, key=lambda p: polar_sort_key(p, anchor))
    
    hull = []
    
    for point in sorted_points:
        # remove points that would create a clockwise turn
        while len(hull) > 1 and cross_product(hull[-2], hull[-1], point) <= 0:
            hull.pop()
        hull.append(point)
    
    return hull


def plot_convex_hull(points, hull, title):
    points = np.array(points)
    hull = np.array(hull)

    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], label='Points')


    closed_hull = np.vstack([hull, hull[0]])
    plt.plot(closed_hull[:, 0], closed_hull[:, 1], 'r-', lw=2, label='Convex Hull')

    plt.title(title)
    plt.axis('equal')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    filename = title.replace(" ", "_") + ".png"
    plt.savefig(filename, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    small_examples = [
        [(0,0), (1,1), (2,2), (3,3)],
        [(0,0), (1,1), (2,0), (1,0.5)],
        [(0,1), (1,2), (2,2), (3,1), (2,0), (1,0)],
    ]

    for i, pts in enumerate(small_examples, start=1):
        hull = graham_scan(pts)
        plot_convex_hull(pts, hull, f"Small Example #{i}")

    sizes = [10, 50, 200, 1000, 3000]

    for n in sizes:
        pts = np.random.rand(n, 2) 
        hull = graham_scan(pts)
        plot_convex_hull(pts, hull, f"Random Example with {n} Points")