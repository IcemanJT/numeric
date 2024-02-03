import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

def natural_cubic_spline(nodes, f_values, f_bis, j, x):
    h = nodes[j+1] - nodes[j]
    a = (nodes[j+1] - x) / h
    b = (x - nodes[j]) / h
    c = (a**3 - a) * h**2 / 6
    d = (b**3 - b) * h**2 / 6
    return a * f_values[j] + b * f_values[j+1] + c * f_bis[j] + d * f_bis[j+1]

if __name__ == "__main__":
    # Wczytanie danych z pliku
    data = np.loadtxt("dane.txt")

    # Podzielenie danych na kolumny x i y
    x = data[:, 0]
    y = data[:, 1]
    
    n = x.shape[0]
    
    x_bis = np.zeros(n)
    
    f = np.array([(y[i] -2*y[i+1] + y[i+2]) for i in range(n-2)])
    h = x[1] - x[0]
    f = f * (6/h**2)
    
    A = np.zeros((n-2, n-2))

    for k in range(n-2):
        for m in range(n-2):
            if k == m:
                A[k, m] = 4
            elif abs(k - m) == 1:
                A[k, m] = 1
                       
    C = np.linalg.cholesky(A)
    z = la.solve_triangular(C, f, lower=True)
    x_bis[1:n-1] = la.solve_triangular(C.T, z, lower=False)
    print(x_bis)

    x_step = 0.01
    
    xs = np.array([])
    ys = np.array([])
    
    for i in range(n-1):
        x_range = np.arange(x[i], x[i+1] + x_step, x_step)
        for x_i in x_range:
            xs = np.append(xs, x_i)
            ys = np.append(ys, natural_cubic_spline(x, y, x_bis, i, x_i))

    # Wykres danych oraz krzywej splotu kubicznego
    plt.scatter(x, y, label='Data nodes', s=5, color='blue')
    plt.plot(xs, ys, label='Cubic spline', color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Natural Cubic Spline')
    plt.show()
