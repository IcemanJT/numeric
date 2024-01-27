import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import baryrat as br


np.set_printoptions(suppress=True, linewidth=100)

def f_x(x):
    return 1/(1+5*x**2)

if __name__ == '__main__':
    
    nodes = np.array([-7/8, -5/8, -3/8, -1/8, 1/8, 3/8, 5/8, 7/8])
    f_values = [f_x(x) for x in nodes]
    
    inter = br.floater_hormann(nodes, f_values, 3)
    
    x_set = np.arange(min(nodes), max(nodes), 0.01)
    y = np.array([])
    for x in x_set:
        y = np.append(y, inter(x))
        
    plt.plot(nodes, f_values, 'o', label='nodes')
    plt.plot(x_set, y, label='Floater-Hormann')
    plt.axis([min(nodes)-.1, max(nodes)+.1, 0.1, 1.1])
    plt.xticks(nodes)
    plt.grid()
    plt.legend()
    plt.show()
    
    
