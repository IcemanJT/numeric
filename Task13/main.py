import numpy as np
import matplotlib.pyplot as plt
import random
import cmath
import scipy.linalg as la

np.set_printoptions(precision=3, suppress=True)

def laguaerre_iteration(poly, z, tolerance=1e-10):
    prev_z = z
    p_1d = poly.deriv()
    p_2d = p_1d.deriv()
    n = poly.o
    while True:
        poly_z = poly(z)
        p_1d_z = p_1d(z)
        p_2d_z = p_2d(z)
        numerator = n * poly_z
        denominator = p_1d_z
        denominator_sqrt = ((n - 1) * ((n - 1) * p_1d_z**2 - n * poly_z * p_2d_z))**(1/2)
        denominator += denominator_sqrt if denominator > 0 else -1 * denominator_sqrt
        z = z - numerator/denominator
        if abs(z - prev_z) < tolerance:
            return z
        prev_z = z
    
def deflate_poly(poly, root):
    a = poly.coef[:-1]
    A = np.eye(len(a)) - np.eye(len(a), k=-1) * root
    b = la.solve_triangular(A, a, lower=True)
    return np.poly1d(b)

def calculate_roots(poly):
    solution = []
    deflated_poly = np.poly1d(poly.coeffs)
    while deflated_poly.o > 2:
        z = random.random()
        root = laguaerre_iteration(deflated_poly, z)
        smooth_root = laguaerre_iteration(poly, root)
        solution.append(smooth_root)
        deflated_poly = deflate_poly(deflated_poly, smooth_root)
        if np.imag(smooth_root) > 1e-4:
            solution.append(np.conj(smooth_root))
            deflated_poly = deflate_poly(deflated_poly, np.conj(smooth_root))
    a, b, c = deflated_poly.coeffs
    delta = b**2 - 4*a*c
    x1 = (-b + cmath.sqrt(delta)) / (2*a)
    x2 = (-b - cmath.sqrt(delta)) / (2*a)
    solution.append(laguaerre_iteration(poly, x1))
    solution.append(laguaerre_iteration(poly, x2))
    return solution
        
if __name__ == '__main__':
    a = np.poly1d(np.array([243, -486, 783, -990, 558, -28, -72, 16], dtype=complex), variable="z", )
    b = np.poly1d(np.array([1, 1, 3, 2, -1, -3, -11, -8, - 12, -4, -4], dtype=complex), variable="z")
    c = np.poly1d(np.array([1, 1j, -1, -1j, 1], dtype=complex), variable="z")
    
    result_a = calculate_roots(a)
    result_b = calculate_roots(b)
    result_c = calculate_roots(c)
    
    print("--------Solution A---------")
    for sol in result_a:
        print(sol)
    print("--------Solution B---------")
    for sol in result_b:
        print(sol)
    print("--------Solution C---------")
    for sol in result_c:
        print(sol)
