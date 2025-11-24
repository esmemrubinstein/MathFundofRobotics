import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import matplotlib.pyplot as plt


def main():
    obstacle_cost = generate_cost()
    gx, gy = np.gradient(obstacle_cost)

    start_point = np.array([10, 10])
    end_point = np.array([90, 90])
    vector = end_point - start_point
    num_pts = 300
    initial_path = start_point + np.outer(np.linspace(0, 1, num_pts), vector)
    
    # Part (a): Obstacle cost only
    # Set to 380 because I earlier ran the convergence version and it took 379 iterations
    # run_optimization(initial_path, obstacle_cost, gx, gy, 
    #                 use_smoothness=False, iterations=[1, 380])
    
    # Part (b and c): Obstacle cost + Smoothness, adjust args for part (b) or (c)
    run_optimization(initial_path, obstacle_cost, gx, gy,
                    use_smoothness=True, iterations=[100, 5000])

def run_optimization(initial_path, cost, gx, gy, use_smoothness=False, iterations=None):
    if iterations is None:
        # Run until convergence - only used for part a
        path, num_iters = optimize_until_convergence(
            initial_path, cost, gx, gy, use_smoothness)
        plot_scene(path, cost, f"Converged after {num_iters} iterations")
    else:
        # Run for specific iteration counts
        for n in iterations:
            path = optimize_path(initial_path, gx, gy, n, use_smoothness)
            plot_scene(path, cost, f"Path After {n} Iterations")


def optimize_path(path, gx, gy, num_iterations, use_smoothness=False,
                 obstacle_weight=0.8, smoothness_weight=4.0, step_size=0.1):
    optimized_path = path.copy()
    
    for iteration in range(num_iterations):
        # obstacle gradient
        obstacle_gradient = compute_obstacle_gradient(optimized_path, gx, gy)
        
        if use_smoothness:
            # smoothness gradient
            smoothness_gradient = compute_smoothness_gradient(optimized_path)
            
            # gradients with weights
            combined_gradient = (obstacle_weight * obstacle_gradient + 
                               smoothness_weight * smoothness_gradient)
        else:
            # only obstacle gradient
            combined_gradient = obstacle_gradient
        
        # keep start and goal fixed
        optimized_path[1:-1] -= step_size * combined_gradient[1:-1]
    
    return optimized_path


def optimize_until_convergence(path, cost, gx, gy, use_smoothness=False,
                               convergence_threshold=0.01, max_iterations=1000):
    optimized_path = path.copy()
    prev_cost = compute_total_cost(optimized_path, cost)
    
    for iteration in range(max_iterations):
        obstacle_gradient = compute_obstacle_gradient(optimized_path, gx, gy)
        
        if use_smoothness:
            pass #N/A
        else:
            combined_gradient = obstacle_gradient
        
        optimized_path[1:-1] -= 0.1 * combined_gradient[1:-1]
        
        current_cost = compute_total_cost(optimized_path, cost)
        if abs(prev_cost - current_cost) < convergence_threshold:
            return optimized_path, iteration + 1
        prev_cost = current_cost
    
    return optimized_path, max_iterations


def compute_obstacle_gradient(path, gx, gy):
    x_coords = np.clip(path[:, 0].astype(int), 0, gx.shape[0] - 1)
    y_coords = np.clip(path[:, 1].astype(int), 0, gx.shape[1] - 1)
    
    grad_x = gx[x_coords, y_coords]
    grad_y = gy[x_coords, y_coords]
    
    return np.column_stack([grad_x, grad_y])


def compute_smoothness_gradient(path):
    smoothness_gradient = np.zeros_like(path)
    # Each point pulls toward its predecessor
    # part b
    # smoothness_gradient[1:] = path[1:] - path[:-1]
    # part c
    smoothness_gradient[1:-1] = 2*path[1:-1] - path[:-2] - path[2:]
    return smoothness_gradient


def compute_total_cost(path, cost):
    values = get_values(path, cost)
    return np.sum(values)


def generate_cost():
    n = 101
    obstacles = np.array([[20, 30], [60, 40], [70, 85]])
    epsilon = np.array([[25], [20], [30]])
    obstacle_cost = np.zeros((n, n))
    
    for i in range(obstacles.shape[0]):
        t = np.ones((n, n))
        t[obstacles[i, 0], obstacles[i, 1]] = 0
        t_cost = distance_transform_edt(t)
        t_cost[t_cost > epsilon[i]] = epsilon[i]
        t_cost = (1 / (2 * epsilon[i])) * (t_cost - epsilon[i])**2
        obstacle_cost += t_cost
    
    return obstacle_cost


def get_values(path, cost):
    x = np.clip(path[:, 0].astype(int), 0, cost.shape[0] - 1)
    y = np.clip(path[:, 1].astype(int), 0, cost.shape[1] - 1)
    return cost[x, y].reshape((path.shape[0], 1))


def plot_scene(path, cost, title):
    values = get_values(path, cost)
    
    # 2D plot
    plt.figure(figsize=(10, 8))
    plt.imshow(cost.T, origin='lower', cmap='hot')
    plt.colorbar(label='Cost')
    plt.plot(path[:, 0], path[:, 1], "ro", markersize=2)
    plt.title(title, fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.tight_layout()
    
    # 3D plot
    fig3d = plt.figure(figsize=(12, 9))
    ax3d = fig3d.add_subplot(111, projection="3d")
    xx, yy = np.meshgrid(range(cost.shape[1]), range(cost.shape[0]))
    ax3d.plot_surface(xx, yy, cost.T, cmap=plt.get_cmap("coolwarm"), alpha=0.7)
    ax3d.scatter(path[:, 0], path[:, 1], values, s=20, c="r")
    ax3d.set_xlabel('X', fontsize=12)
    ax3d.set_ylabel('Y', fontsize=12)
    ax3d.set_zlabel('Cost', fontsize=12)
    ax3d.set_title(title + " (3D View)", fontsize=14)
    
    plt.show()


if __name__ == "__main__":
    main()