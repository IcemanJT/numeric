import numpy as np
import math


# szukam e^(-A) < 1e-7
def find_limit(tolerance=1e-7):
    A = 0
    while (math.exp(-A) > tolerance):
        A += 1
    return A

# funkcja podcałkowa
def f_x(x):
    return math.sin(math.pi * (1.0 + math.sqrt(x)) / (1.0 + x**2)) * math.exp(-x)

# kryterium stopu
def intergral_tolerance(approx_new, approx_old, tolerance=1e-7, epsilon=1e-4):
    if approx_old is None:
        return False
    return abs(approx_new - approx_old) / (abs(approx_old) + epsilon) < tolerance

def calculate_integral(x_start, x_end):
        prev_approx = None
        integral_sum = 0.5 * (f_x(x_start) + f_x(x_end))
        n = 2
        
        while True:
            xs, h = np.linspace(x_start, x_end, n, retstep=True)
            for x in xs[1::2]:
                integral_sum += f_x(x)
            approximation = integral_sum * h
            if intergral_tolerance(approximation, prev_approx):
                break
            prev_approx = approximation
            n += (n - 1)
            
            
        return approximation
    


x_start = 0.0
x_end = find_limit()

result = calculate_integral(x_start, x_end)
print(result)
