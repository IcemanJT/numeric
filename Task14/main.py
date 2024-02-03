import numpy as np
import matplotlib.pyplot as plt
import random

np.set_printoptions(suppress=True)

MAX_SOLUTIONS = 4
MAX_ITERATIONS = 10000
MAX_GUESSES = 1000


f0 = lambda x, y: 2 * x**2 + y**2 - 2
f1 = lambda x, y: (x - 1/2)**2 + (y - 1)**2 - 1/4
f0_dx = lambda x: 4*x
f0_dy = lambda y: 2*y
f1_dx = lambda x: 2*(x - 1/2)
f1_dy = lambda y: 2 * (y - 1)


def generate_num():
    x = random.random() * random.randint(1, 3)
    
    make_x_imag = random.random()
    x = x * 1j if make_x_imag > 0.5 else x
    
    negate_x = random.random()
    x = x * -1 if negate_x > 0.5 else x
    
    return x

def newton_damped_method():
    w = 1
    w_shrunk = False
    x = generate_num()
    y = generate_num()
    for _ in range(MAX_ITERATIONS):
        if w < 1e-6:
            return None
        if not w_shrunk:
            jac = np.array([[f0_dx(x), f0_dy(y)],[f1_dx(x), f1_dy(y)]], dtype=complex)
            g_x = np.array([[f0(x, y)], [f1(x, y)]], dtype=complex)
            delta_x = np.linalg.solve(-1 * jac, g_x)
        x_n = x + w * delta_x[0][0]
        y_n = y + w * delta_x[1][0]
        grad_x = 1/2 * np.linalg.norm(g_x)
        g_n = np.array([[f0(x_n, y_n)], [f1(x_n, y_n)]], dtype=complex)
        grad_x_n = 1/2 * np.linalg.norm(g_n)
        if abs(grad_x - grad_x_n) < 1e-8:
            return x_n, y_n
        if grad_x_n < grad_x:
            w = 1
            w_shrunk = False
            x = x_n
            y = y_n
        else:
            w_shrunk = True
            w /= 2    
    


if __name__ == "__main__":
    solutions = []
    for _ in range(MAX_GUESSES):
        is_close = False
        if len(solutions) == MAX_SOLUTIONS:
            break
        (x, y) = newton_damped_method()
        solution = np.array([x, y], dtype=complex)
        for sol in solutions:
            if np.allclose(sol, solution):
                is_close = True
                break
        if not is_close:
            solutions.append(solution)
        
    print(solutions)
