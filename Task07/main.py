import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from math import *

np.set_printoptions(suppress=True, linewidth=130)

# calculates coefficients using lagraunge interpolation method
def lagrange_interpolation(nodes, nodes_values, precision=1e-8):
    n = len(nodes)
    interpolated_polynomial = np.poly1d([0.0])  # Initialize the interpolated polynomial


    for i in range(n):
        numerator = np.poly1d([1.0])    # licznik lj(x)
        denominator = 1.0               # mianownika lj(x)           

        for j in range(n):
            if i != j:
                numerator *= np.poly1d([1.0, -nodes[j]])    # obliczanie licznika wielomianu lj(x)
                denominator *= nodes[i] - nodes[j]          # obliczanie mianownika wielomianu lj(x)
        
        # one operation = one iteration
        
        interpolated_polynomial += nodes_values[i] * (numerator / denominator)

    return interpolated_polynomial


if __name__ == '__main__':

    nodes = np.array([-0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00])
    nodes_values = np.array([1.1309204101562500, 2.3203125000000000, 1.9284057617187500, 1.0000000000000000, 0.0554809570312500, -0.6015625000000000, -0.7525024414062500, 0.0000000000000000])
    
    poly = lagrange_interpolation(nodes, nodes_values)
    
    print(poly)
    
    
    x_start = -1.25
    x_end = 1.25
    x_step = 0.01
    
    x = np.arange(x_start, x_end, x_step)
    y = np.array([poly(x_i) for x_i in x])
    

    plt.plot(nodes, nodes_values, 'o', color='red')
    plt.plot(x, y, 'green')
    plt.axis([-1.5, 1.5, -20, 3.0])
    plt.xticks(np.arange(-1.5, 1.5, 0.25))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Plot of f(x)')
    plt.grid(True)
    plt.show()
