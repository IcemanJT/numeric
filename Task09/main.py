import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

np.set_printoptions(suppress=True, linewidth=100)

def f_x(x):
    return 1/(1+5*x**2)

    
    
def natural_cubic_spline(nodes, f_values, f_bis, j, x):
    h = nodes[j+1] - nodes[j]
    a = (nodes[j+1] - x) / h
    b = (x - nodes[j]) / h
    c = (a**3 - a) * h**2 / 6
    d = (b**3 - b) * h**2 / 6
    return a * f_values[j] + b * f_values[j+1] + c * f_bis[j] + d * f_bis[j+1]


if __name__ == '__main__':
    
    nodes = np.array([-7/8, -5/8, -3/8, -1/8, 1/8, 3/8, 5/8, 7/8])
    f_values = [f_x(x) for x in nodes]
    
    n = nodes.shape[0]
    
    x_bis = np.zeros(n)
    
    f = np.array([(f_values[i] -2*f_values[i+1] + f_values[i+2]) for i in range(n-2)])
    h = 1/4
    f = f * (6/h**2)
    
    A = np.zeros((n-2, n-2))

    for k in range(n-2):
        for m in range(n-2):
            if k == m:
                A[k, m] = 4
            elif abs(k - m) == 1:
                A[k, m] = 1
                
    print(A)            
    C = np.linalg.cholesky(A)
    y = la.solve_triangular(C, f, lower=True)
    x_bis[1:n-1] = la.solve_triangular(C.T, y, lower=False)

    x_step = 0.01
    
    xs = np.array([])
    ys = np.array([])
    
    for i in range(n-1):
        x_range = np.arange(nodes[i], nodes[i+1] + x_step, x_step)
        for x in x_range:
            xs = np.append(xs, x)
            ys = np.append(ys, natural_cubic_spline(nodes, f_values, x_bis, i, x))
    

    plt.plot(nodes, f_values, 'o', label='data')
    plt.plot(xs, ys, label='spline')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.axis([min(nodes)-.1, max(nodes)+.1, 0.1, 1])
    plt.xticks(np.arange(min(nodes), max(nodes)+0.1, 0.25))
    plt.grid()
    plt.legend()
    plt.show()
    
    
    
    
