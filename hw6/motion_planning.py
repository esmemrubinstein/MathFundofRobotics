import numpy as np
from convex_hull import graham_scan
from shortest_paths import find_shortest_path
import matplotlib.pyplot as plt

EPS = 1e-9

def negate_polygon(polygon):
    return [(-p[0], -p[1]) for p in polygon]


def normalize_angle(angle):
    """normalize angle to [-pi, pi]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def get_edge_angle(v1, v2):
    dx = v2[0] - v1[0]
    dy = v2[1] - v1[1]
    return np.arctan2(dy, dx)


def minkowski_difference(obstacle, robot):
    # Get reference point and translate robot so reference point is at origin
    ref_point = get_robot_reference_point(robot)
    robot_at_origin = [(v[0] - ref_point[0], v[1] - ref_point[1]) for v in robot]
    
    neg_robot = negate_polygon(robot_at_origin)
    
    combined = []
    for obs_v in obstacle:
        for rob_v in neg_robot:
            combined.append((obs_v[0] + rob_v[0], obs_v[1] + rob_v[1]))
    
    cspace_obstacle = graham_scan(combined)
    
    return cspace_obstacle


def compute_configuration_space(obstacles, robot):
    cspace_obstacles = []
    
    for obstacle in obstacles:
        cspace_obs = minkowski_difference(obstacle, robot)
        cspace_obstacles.append(cspace_obs)
    
    return cspace_obstacles


def get_robot_reference_point(robot):
    return robot[0]


def translate_robot(robot, reference_point, new_position):
    dx = new_position[0] - reference_point[0]
    dy = new_position[1] - reference_point[1]
    
    return [(v[0] + dx, v[1] + dy) for v in robot]


def plan_robot_motion(robot, obstacles, start_config, goal_config):
    cspace_obstacles = compute_configuration_space(obstacles, robot)
    path, length = find_shortest_path(cspace_obstacles, start_config, goal_config)
    
    return path, length


def visualize_robot_path(robot, obstacles, start_config, goal_config, 
                         path=None, num_robot_positions=8, title="Robot Motion Planning"):

    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw obstacles
    for obs in obstacles:
        obs_arr = np.array(obs)
        closed = np.vstack([obs_arr, obs_arr[0]])
        ax.fill(closed[:, 0], closed[:, 1], 
                alpha=0.4, color='lightcoral', edgecolor='darkred', linewidth=2.5,
                label='Obstacle' if obs == obstacles[0] else '')
    
    ref_point = get_robot_reference_point(robot)
    
    if path is not None:
        path_arr = np.array(path)
        ax.plot(path_arr[:, 0], path_arr[:, 1], 
               'b-', linewidth=2.5, alpha=0.7, label='Path', zorder=3)
        
        ax.scatter(path_arr[:, 0], path_arr[:, 1],
                  c='blue', s=80, alpha=0.6, zorder=4, label='Waypoints')
        
        if len(path) > 2:
            indices = np.linspace(1, len(path) - 2, 
                                min(num_robot_positions, len(path) - 2),
                                dtype=int)
            
            for idx in indices:
                pos = path[idx]
                robot_at_pos = translate_robot(robot, ref_point, pos)
                robot_arr = np.array(robot_at_pos)
                closed_robot = np.vstack([robot_arr, robot_arr[0]])
                ax.fill(closed_robot[:, 0], closed_robot[:, 1],
                       alpha=0.15, color='gray', edgecolor='gray', 
                       linewidth=1, linestyle='--')
    
    start_robot = translate_robot(robot, ref_point, start_config)
    start_arr = np.array(start_robot)
    closed_start = np.vstack([start_arr, start_arr[0]])
    ax.fill(closed_start[:, 0], closed_start[:, 1],
           alpha=0.5, color='lightgreen', edgecolor='darkgreen', 
           linewidth=2.5, label='Start Config')
    ax.scatter(*start_config, c='green', s=250, marker='o',
              edgecolors='darkgreen', linewidth=2.5, zorder=5)
    
    goal_robot = translate_robot(robot, ref_point, goal_config)
    goal_arr = np.array(goal_robot)
    closed_goal = np.vstack([goal_arr, goal_arr[0]])
    ax.fill(
        closed_goal[:, 0], closed_goal[:, 1],
        alpha=0.5,
        color='#D8B4FE',          # light purple fill
        edgecolor='#5B21B6',      # dark purple outline
        linewidth=2.5,
        label='Goal Config'
    )

    ax.scatter(
        *goal_config,
        c='#7C3AED',              # medium purple star
        s=300,
        marker='*',
        edgecolors='#5B21B6',     # dark purple outline
        linewidth=2.5,
        zorder=5
    )
    
    if path is None:
        ax.text(0.5, 0.95, "NO PATH EXISTS", 
               transform=ax.transAxes, fontsize=20, color='red',
               ha='center', va='top', weight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    else:
        ax.text(0.5, 0.95, f"Path Length: {path[1]:.2f}" if isinstance(path, tuple) else f"Path found with {len(path)} waypoints",
               transform=ax.transAxes, fontsize=14, color='darkblue',
               ha='center', va='top', weight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.show()


def run_motion_planning_tests():
    # Test 1: Square robot, single square obstacle
    print("=" * 60)
    print("Test 1: Square Robot - Single Square Obstacle")
    print("=" * 60)
    robot1 = [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]
    obstacles1 = [[(3, 3), (5, 3), (5, 5), (3, 5)]]
    start1 = (1, 1)
    goal1 = (7, 7)
    
    path1, length1 = plan_robot_motion(robot1, obstacles1, start1, goal1)
    print(f"Path found: {path1 is not None}")
    if path1:
        print(f"Path length: {length1:.2f}")
        print(f"Number of waypoints: {len(path1)}")
    visualize_robot_path(robot1, obstacles1, start1, goal1, path1,
                        title="Test 1: Single Square Obstacle")
    print()
    
    # Test 2: Triangle robot, multiple obstacles (maze-like)
    print("=" * 60)
    print("Test 2: Triangle Robot - Maze Environment")
    print("=" * 60)
    robot2 = [(0, 0), (0.6, 0), (0.3, 0.5)]
    obstacles2 = [
        [(1, 2), (4, 2), (4, 3), (1, 3)],
        [(2, 4), (5, 4), (5, 5), (2, 5)],
        [(3, 6), (6, 6), (6, 7), (3, 7)],
        [(5, 1), (8, 1), (8, 2), (5, 2)],
    ]
    start2 = (0.5, 0.5)
    goal2 = (7, 8)
    
    path2, length2 = plan_robot_motion(robot2, obstacles2, start2, goal2)
    print(f"Path found: {path2 is not None}")
    if path2:
        print(f"Path length: {length2:.2f}")
        print(f"Number of waypoints: {len(path2)}")
    visualize_robot_path(robot2, obstacles2, start2, goal2, path2,
                        title="Test 2: Maze Environment")
    print()
    
    # Test 3: Narrow passage
    print("=" * 60)
    print("Test 3: Square Robot - Narrow Passage")
    print("=" * 60)
    robot3 = [(0, 0), (0.4, 0), (0.4, 0.4), (0, 0.4)]
    obstacles3 = [
        [(2, 0), (2, 3), (2.5, 3), (2.5, 0)],  # Left wall
        [(2, 5), (2, 8), (2.5, 8), (2.5, 5)],  # Left wall top
        [(4.5, 0), (4.5, 3), (5, 3), (5, 0)],  # Right wall
        [(4.5, 5), (4.5, 8), (5, 8), (5, 5)],  # Right wall top
    ]
    start3 = (0.5, 4)
    goal3 = (6.5, 4)
    
    path3, length3 = plan_robot_motion(robot3, obstacles3, start3, goal3)
    print(f"Path found: {path3 is not None}")
    if path3:
        print(f"Path length: {length3:.2f}")
        print(f"Number of waypoints: {len(path3)}")
    visualize_robot_path(robot3, obstacles3, start3, goal3, path3,
                        title="Test 3: Narrow Passage")
    print()
    
    # Test 4: No path (goal enclosed)
    print("=" * 60)
    print("Test 4: No Path - Enclosed Goal")
    print("=" * 60)
    robot4 = [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]
    obstacles4 = [
        [(4, 4), (8, 4), (8, 8), (4, 8)]  # Box enclosing goal
    ]
    start4 = (1, 1)
    goal4 = (6, 6)
    
    path4, length4 = plan_robot_motion(robot4, obstacles4, start4, goal4)
    print(f"Path found: {path4 is not None}")
    if path4:
        print(f"Path length: {length4:.2f}")
    visualize_robot_path(robot4, obstacles4, start4, goal4, path4,
                        title="Test 4: No Path (Enclosed Goal)")
    print()
    
    # Test 5: Complex environment
    print("=" * 60)
    print("Test 5: Hexagon Robot - Complex Environment")
    print("=" * 60)
    # Hexagon robot
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    robot5 = [(0.3*np.cos(a), 0.3*np.sin(a)) for a in angles]
    
    obstacles5 = [
        [(1, 1), (3, 1), (3, 2), (1, 2)],
        [(4, 2), (6, 2), (6, 4), (4, 4)],
        [(1, 5), (2, 5), (2, 7), (1, 7)],
        [(3, 6), (5, 6), (5, 7), (3, 7)],
        [(6, 5), (8, 5), (8, 7), (6, 7)],
        [(2, 3), (3, 3), (3, 4), (2, 4)],
    ]
    start5 = (0.5, 0.5)
    goal5 = (8.5, 8.5)
    
    path5, length5 = plan_robot_motion(robot5, obstacles5, start5, goal5)
    print(f"Path found: {path5 is not None}")
    if path5:
        print(f"Path length: {length5:.2f}")
        print(f"Number of waypoints: {len(path5)}")
    visualize_robot_path(robot5, obstacles5, start5, goal5, path5,
                        num_robot_positions=10,
                        title="Test 5: Complex Environment")
    print()


def test_overlapping_cspace_obstacles():
    print("=" * 60)
    print("Test: Overlapping C-space Obstacles")
    print("=" * 60)
    
    # Square robot
    robot = [(0, 0), (0.6, 0), (0.6, 0.6), (0, 0.6)]
    
    # Two small obstacles close together
    # In workspace they don't touch, but in C-space they will overlap
    obstacles = [
        [(3, 3), (4, 3), (4, 4), (3, 4)],
        [(4.5, 3), (5.5, 3), (5.5, 4), (4.5, 4)]
    ]
    
    start = (1, 3.5)
    goal = (7, 3.5)

    cspace_obstacles = compute_configuration_space(obstacles, robot)
    
    print(f"Original obstacle 1: {obstacles[0]}")
    print(f"Original obstacle 2: {obstacles[1]}")
    print(f"C-space obstacle 1: {len(cspace_obstacles[0])} vertices")
    print(f"C-space obstacle 2: {len(cspace_obstacles[1])} vertices")
    print("\nNote: The C-space obstacles will overlap because the robot is 0.6 units wide")
    print("and the gap between obstacles is only 0.5 units.")
    

    path, length = plan_robot_motion(robot, obstacles, start, goal)
    print(f"\nPath found: {path is not None}")
    if path:
        print(f"Path length: {length:.2f}")
        print(f"Number of waypoints: {len(path)}")
        print("The robot navigates around the combined forbidden region created by overlapping C-space obstacles.")
    
    visualize_robot_path(robot, obstacles, start, goal, path,
                        num_robot_positions=6,
                        title="Overlapping C-space Obstacles")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.set_title("Workspace (Original)", fontsize=14, weight='bold')
    for obs in obstacles:
        obs_arr = np.array(obs)
        closed = np.vstack([obs_arr, obs_arr[0]])
        ax1.fill(closed[:, 0], closed[:, 1], 
                alpha=0.4, color='lightcoral', edgecolor='darkred', linewidth=2)
    
    ref_point = robot[0]
    start_robot = translate_robot(robot, ref_point, start)
    start_arr = np.array(start_robot)
    closed_start = np.vstack([start_arr, start_arr[0]])
    ax1.fill(closed_start[:, 0], closed_start[:, 1],
            alpha=0.5, color='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax1.scatter(*start, c='green', s=200, marker='o', zorder=5)
    
    goal_robot = translate_robot(robot, ref_point, goal)
    goal_arr = np.array(goal_robot)
    closed_goal = np.vstack([goal_arr, goal_arr[0]])
    ax1.fill(closed_goal[:, 0], closed_goal[:, 1],
            alpha=0.5, color='#D8B4FE', edgecolor='#5B21B6', linewidth=2)
    ax1.scatter(*goal, c='#7C3AED', s=300, marker='*', zorder=5)
    
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 8)
    ax1.set_ylim(2, 5)
    
    ax2.set_title("Configuration Space (C-space obstacles OVERLAP)", fontsize=14, weight='bold')
    colors = ['lightcoral', 'lightblue']
    for i, cspace_obs in enumerate(cspace_obstacles):
        obs_arr = np.array(cspace_obs)
        closed = np.vstack([obs_arr, obs_arr[0]])
        ax2.fill(closed[:, 0], closed[:, 1],
                alpha=0.4, color=colors[i], edgecolor='darkred', linewidth=2,
                label=f'C-space Obstacle {i+1}')
    
    ax2.scatter(*start, c='green', s=200, marker='o', zorder=5, label='Start')
    ax2.scatter(*goal, c='#7C3AED', s=300, marker='*', zorder=5, label='Goal')
    
    if path:
        path_arr = np.array(path)
        ax2.plot(path_arr[:, 0], path_arr[:, 1], 'b-', linewidth=2.5, label='Path', zorder=3)
    
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2.set_xlim(0, 8)
    ax2.set_ylim(2, 5)
    
    plt.tight_layout()
    plt.savefig("Overlapping_CSpace_Comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nExplanation:")
    print("- In the workspace (left), two obstacles are separated by 0.5 units")
    print("- The robot is 0.6 units wide, so it cannot fit through the gap")
    print("- In C-space (right), the grown obstacles overlap, creating one merged forbidden region")
    print("- The visibility graph correctly treats this as impassable")
    print("- The robot must go around (above or below) rather than between the obstacles")


if __name__ == "__main__":
    run_motion_planning_tests()
    test_overlapping_cspace_obstacles()