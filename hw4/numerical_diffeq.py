import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def exact_solution(x):
    return np.cbrt(x)


def f(x, y):
    return 1 / (3 * y**2)


def euler_backward(h=0.05):
    # Start from x=1, go backward to x=0
    x = np.arange(1.0, -h/2, -h)
    y = np.zeros(len(x))
    
    y[0] = 1.0
    
    for i in range(len(x) - 1):
        y[i+1] = y[i] - h * f(x[i], y[i])
    
    x = x[::-1]
    y = y[::-1]
    
    return x, y


def rk4_backward(h=0.05):
    # Start from x=1, go backward to x=0
    x = np.arange(1.0, -h/2, -h)
    y = np.zeros(len(x))
    
    y[0] = 1.0
    
    for i in range(len(x) - 1):
        x_n = x[i]
        y_n = y[i]
        
        k1 = f(x_n, y_n)
        k2 = f(x_n - h/2, y_n - (h/2)*k1)
        k3 = f(x_n - h/2, y_n - (h/2)*k2)
        k4 = f(x_n - h, y_n - h*k3)
        
        y[i+1] = y_n - (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    x = x[::-1]
    y = y[::-1]
    
    return x, y


def adams_bashforth4_backward(h=0.05):
    # Start from x=1.15 and go backward to x=0
    x_start = 1.15
    x = np.arange(x_start, -h/2, -h)
    y = np.zeros(len(x))
    
    # Given initial values (in backward order from x=1.15)
    y[0] = 1.04768955317165  # y(1.15)
    y[1] = 1.03228011545637  # y(1.10)
    y[2] = 1.01639635681485  # y(1.05)
    y[3] = 1.00000000000000  # y(1.00)
    
    f_vals = [f(x[i], y[i]) for i in range(4)]
    
    for i in range(3, len(x) - 1):
        f_n = f_vals[-1]
        f_n1 = f_vals[-2]
        f_n2 = f_vals[-3]
        f_n3 = f_vals[-4]
        
        y[i+1] = y[i] - (h/24) * (55*f_n - 59*f_n1 + 37*f_n2 - 9*f_n3)
        
        f_new = f(x[i+1], y[i+1])
        f_vals.append(f_new)
        
        if len(f_vals) > 4:
            f_vals.pop(0)
    
    x = x[::-1]
    y = y[::-1]
    
    mask = x <= 1.0
    x = x[mask]
    y = y[mask]
    
    return x, y


def create_comparison_table(h=0.05):
    x_euler, y_euler = euler_backward(h)
    x_rk4, y_rk4 = rk4_backward(h)
    x_ab4, y_ab4 = adams_bashforth4_backward(h)
    
    # Use x from Euler (they should all be the same for [0,1])
    x = x_euler
    y_exact = exact_solution(x)
    
    df = pd.DataFrame({
        'x': x,
        'y_exact': y_exact,
        'y_euler': y_euler,
        'error_euler': y_exact - y_euler,
        'y_rk4': y_rk4,
        'error_rk4': y_exact - y_rk4,
        'y_ab4': y_ab4,
        'error_ab4': y_exact - y_ab4
    })
    
    return df


def plot_solutions(h=0.05):
    x_euler, y_euler = euler_backward(h)
    x_rk4, y_rk4 = rk4_backward(h)
    x_ab4, y_ab4 = adams_bashforth4_backward(h)
    
    x_fine = np.linspace(0, 1, 1000)
    y_fine = exact_solution(x_fine)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    
    ax1.plot(x_fine, y_fine, 'k-', linewidth=2, label='Exact Solution', zorder=1)
    ax1.plot(x_euler, y_euler, 'ro-', markersize=4, label='Euler', alpha=0.7, zorder=2)
    ax1.plot(x_rk4, y_rk4, 'bs-', markersize=4, label='RK4', alpha=0.7, zorder=3)
    ax1.plot(x_ab4, y_ab4, 'g^-', markersize=4, label='Adams-Bashforth 4', alpha=0.7, zorder=4)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Numerical Solutions vs Exact Solution', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    plt.savefig('ode_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    h = 0.05
    df = create_comparison_table(h)
    
    # Display table
    print("COMPARISON TABLE")
    print("-" * 80)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.10f}'.format)
    print(df.to_string(index=False))
    print()
    
    plot_solutions(h)

if __name__ == "__main__":
    main()