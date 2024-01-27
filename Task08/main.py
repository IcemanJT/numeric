import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, linewidth=100)

def f_x(x):
    return 1/(1+5*x**2)

def lagrange_interpolation(nodes, nodes_values):
    n = len(nodes)
    coefficients = np.zeros(n)

    for i in range(n):
        numerator = np.poly1d([1.0])
        denominator = 1.0

        for j in range(n):
            if i != j:
                numerator *= np.poly1d([1.0, -nodes[j]])
                denominator *= nodes[i] - nodes[j]

        coefficients += nodes_values[i] * (numerator / denominator)

    return coefficients

if __name__ == '__main__':
    
    nodes = [-7/8, -5/8, -3/8, -1/8, 1/8, 3/8, 5/8, 7/8]
    values = [f_x(x) for x in nodes]
    
    # Lagrange interpolation
    x = np.linspace(min(nodes),max(nodes) , 100)
    y = np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(nodes)):
            y[i] += values[j]*np.prod([(x[i]-nodes[k])/(nodes[j]-nodes[k]) for k in range(len(nodes)) if k!=j])
            
    # Coefficients
    c = lagrange_interpolation(nodes, values)
    print(c)
            
    plt.plot(x, y, label='Lagrange interpolation')
    plt.scatter(nodes, values, label='Nodes', color='red')
    plt.axis([min(nodes)-.1, max(nodes)+0.1, 0.1, 1])
    plt.xticks(np.arange(min(nodes), max(nodes)+0.1, 0.25))
    plt.grid()
    plt.legend()
    plt.show()
    
