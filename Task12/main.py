import numpy as np
import math
import matplotlib.pyplot as plt

STACK_CAPACITY = 1000

def find_limit(tolerance=1e-8):
    x = 0
    while math.exp(-(x**2)) > tolerance:
        x += 1
    return x 

def f_x(t):
    return math.cos((1 + t) / (t**2 + 0.04)) * math.exp(-(t**2))

def calculate_error(result_left, result_right, result, tolerance=1e-8):
    return  abs((result_left + result_right - result) / 3)  < tolerance

def trapezoidal_rule_integration(x_start, x_end):
    return (x_end - x_start)/2 * (f_x(x_start) + f_x(x_end))

def calculate_trapezoidal_approx(left, right, all):
    return (4 *(right + left) - all) / 3


if __name__ == "__main__":
    
    x_end = find_limit()
    x_start = x_end* -1
    x_mid = (x_start + x_end) / 2
    
    integral = trapezoidal_rule_integration(x_start, x_end)
    result = 0
    
    stack = []
    xs = []
    ys = []
    
    while(True):
        integral_left = trapezoidal_rule_integration(x_start, x_mid)
        integral_right = trapezoidal_rule_integration(x_mid, x_end)
        
        if not calculate_error(integral_left, integral_right, integral):
            stack.append((x_mid, x_end, integral_right))
            x_end = x_mid
            x_mid = (x_start + x_end) / 2
            integral= integral_left
        else:
            xs.append(x_mid) 
            result += calculate_trapezoidal_approx(integral_left, integral_right, integral)
            ys.append(result)
            if not stack:
                break
            (x_start, x_end, integral) = stack.pop()
            x_mid = (x_start + x_end) / 2
        if len(stack) >= STACK_CAPACITY:
            print("Stack overflow.")
            break

    print(result)
    xs0 = np.linspace(-3, 3, 1000)
    ys0 = [f_x(x) for x in xs0]
    plt.title("")
    plt.plot(xs0, ys0, label="f(x)")
    plt.plot(xs, ys, label="F(x)")
    plt.xticks(np.arange(-3, 3+.01, 1))
    plt.legend()
    plt.grid("both")
    plt.show()
    