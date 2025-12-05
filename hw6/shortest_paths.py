# Improved shortest paths in 2D polygonal environments
# Cleaner structure with better separation of concerns

import numpy as np
import matplotlib.pyplot as plt
import heapq
from convex_hull import cross_product as cross_2d, distance

EPS = 1e-9


def point_on_segment(p, a, b, eps=EPS):
    """true if p is on segment ab (including endpoints)"""
    cross = abs(cross_2d(a, p, b))
    if cross > eps:
        return False
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)


def segments_intersect_properly(a, b, c, d, eps=EPS):
    """
    true if segments ab and cd intersect in their interiors (not at endpoints).
    """
    cross1 = cross_2d(a, b, c)
    cross2 = cross_2d(a, b, d)
    cross3 = cross_2d(c, d, a)
    cross4 = cross_2d(c, d, b)
    
    # segments properly intersect if endpoints are on opposite sides
    return cross1 * cross2 < -eps and cross3 * cross4 < -eps


def point_in_polygon_strict(p, poly, eps=EPS):
    """True if p is strictly inside convex polygon (not on boundary)"""
    n = len(poly)
    for i in range(n):
        # all cross products must be positive for CCW polygon
        if cross_2d(poly[i], poly[(i+1) % n], p) <= eps:
            return False
    return True


def is_visible(a, b, polygons, eps=EPS):
    """
    returns True if segment ab is a valid visibility edge.
    - Endpoints can touch polygon boundaries/vertices
    - Interior of segment must not intersect any polygon
    """
    a = np.array(a)
    b = np.array(b)
    
    for poly in polygons:
        # if both endpoints inside,  blocked
        if point_in_polygon_strict(a, poly, eps) or point_in_polygon_strict(b, poly, eps):
            return False
        
        n = len(poly)
        for i in range(n):
            v1 = poly[i]
            v2 = poly[(i+1) % n]
            
            if segments_intersect_properly(a, b, v1, v2, eps):
                return False
        
        # sample points along segment interior
        # catches cases where segment might thread through polygon
        num_samples = 10
        for j in range(1, num_samples):
            t = j / num_samples
            sample = a + t * (b - a)
            if point_in_polygon_strict(sample, poly, eps):
                return False
    
    return True


def build_visibility_graph(polygons, start, goal, eps=EPS):
    nodes = [tuple(start), tuple(goal)]
    
    for poly in polygons:
        for vertex in poly:
            v = tuple(vertex)
            # exclude vertices that are strictly inside another polygon
            if not any(point_in_polygon_strict(v, other, eps) for other in polygons):
                if v not in nodes:   
                    nodes.append(v)
    n = len(nodes)
    graph = {i: [] for i in range(n)}
    
    for i in range(n):
        for j in range(i + 1, n):
            if is_visible(nodes[i], nodes[j], polygons, eps):
                d = distance(nodes[i], nodes[j])
                graph[i].append((j, d))
                graph[j].append((i, d))
    
    return nodes, graph

def find_shortest_path(polygons, start, goal, eps=EPS):
    nodes, graph = build_visibility_graph(polygons, start, goal, eps)
    
    # dijkstra's algorithm
    n = len(nodes)
    dist = [float('inf')] * n
    prev = [None] * n
    visited = [False] * n
    
    start_idx = 0  
    goal_idx = 1
    
    dist[start_idx] = 0
    heap = [(0, start_idx)]
    
    while heap:
        d, u = heapq.heappop(heap)
        
        if visited[u]:
            continue
        visited[u] = True
        
        if u == goal_idx:
            break
        
        for v, weight in graph[u]:
            if not visited[v]:
                alt = dist[u] + weight
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(heap, (alt, v))
    
    if dist[goal_idx] == float('inf'):
        return None, None
    
    path = []
    u = goal_idx
    while u is not None:
        path.append(nodes[u])
        u = prev[u]
    path.reverse()
    
    return path, dist[goal_idx]


def plot_environment(polygons, start, goal, path=None, nodes=None, title="Shortest Path"):
    fig, ax = plt.subplots(figsize=(10, 10))

    for poly in polygons:
        poly_arr = np.array(poly)
        closed = np.vstack([poly_arr, poly_arr[0]])
        ax.fill(closed[:, 0], closed[:, 1], 
                alpha=0.3, color='lightcoral', edgecolor='darkred', linewidth=2)
    

    if nodes:
        nodes_arr = np.array(nodes)
        ax.scatter(nodes_arr[:, 0], nodes_arr[:, 1], 
                  c='orange', s=100, zorder=3, alpha=0.6, label='Graph Nodes')
    
    ax.scatter(*start, c='green', s=200, marker='o', 
              edgecolors='darkgreen', linewidth=2, zorder=5, label='Start')
    ax.scatter(*goal, c='red', s=200, marker='*', 
              edgecolors='darkred', linewidth=2, zorder=5, label='Goal')
    
    # Draw path
    if path:
        path_arr = np.array(path)
        ax.plot(path_arr[:, 0], path_arr[:, 1], 
               'b-', linewidth=3, zorder=4, label='Shortest Path')

    else:
        ax.text(0.5, 0.5, "NO PATH EXISTS", 
               transform=ax.transAxes, fontsize=20, color='red',
               ha='center', va='center', weight='bold')
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title(title, fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.show()


def run_tests():
    test_cases = [
        {
            "name": "Single Square Obstacle",
            "polygons": [[(2,2), (5,2), (5,5), (2,5)]],
            "start": (0, 0),
            "goal": (7, 7)
        },
        {
            "name": "Maze Layout",
            "polygons": [
                [(1,1), (4,1), (4,2), (1,2)],
                [(2,3), (5,3), (5,4), (2,4)],
                [(3,5), (6,5), (6,6), (3,6)],
            ],
            "start": (0, 0),
            "goal": (7, 7)
        },
        {
            "name": "Two Triangles",
            "polygons": [
                [(1,1), (3,1), (2,3)],
                [(4,0), (6,0), (5,2)]
            ],
            "start": (0, 0),
            "goal": (7, 1)
        },
        {
            "name": "No Path (Enclosed)",
            "polygons": [[(1,1), (10,1), (10,10), (1,10)]],
            "start": (0, 0),
            "goal": (7, 7)
        }
    ]
    
    for case in test_cases:
        path, length = find_shortest_path(case['polygons'], case['start'], case['goal'])
        nodes, _ = build_visibility_graph(case['polygons'], case['start'], case['goal'])
        plot_environment(case['polygons'], case['start'], case['goal'], 
                        path=path, nodes=nodes, title=case['name'])


if __name__ == "__main__":
    run_tests()
