# Jeremi Toroj 11.11.2023

import numpy as np

"""

Largest Eigenvalue: 4.00000002483527
Corresponding Eigenvector: [0.40824828 0.40824829 0.40824828 0.40824828 0.40824829 0.40824832]
Second Largest Eigenvalue: 2.999999999996387
Corresponding Eigenvector: [-7.07106401e-01  1.05605098e-07 -1.29601208e-08 -1.29601208e-08
 -9.17690458e-07  7.07107161e-01]
 
4.0
[-0.4082483  -0.4082483  -0.40824828 -0.40824828 -0.4082483  -0.4082483 ]
3.0
[-7.0710677e-01 -7.3375844e-16 -1.1891024e-16 -4.2719148e-16
 -4.3895804e-16  7.0710677e-01]
 
"""


def power_method(a, num_iter=10000, tol=1e-24):
    n, d = a.shape
    x = np.random.rand(d)
    x = x / np.linalg.norm(x)
    for _ in range(num_iter):
        x_new = np.dot(a, x)
        x_new = x_new / np.linalg.norm(x_new)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    eigenvalue = np.dot(x, np.dot(a, x)) / np.dot(x, x)
    return eigenvalue, x


def find_second_largest_eigenvalue(a, e1, num_iter=10000, tol=1e-24):
    y1 = np.random.rand(a.shape[1])
    y1 -= y1.dot(e1) * e1
    y1 /= np.linalg.norm(y1)

    for _ in range(num_iter):
        z = np.dot(a, y1)
        z -= e1 * np.dot(e1, z)  # Making z orthogonal to e1
        z /= np.linalg.norm(z)   # Normalizing z

        if np.allclose(y1, z, atol=tol):   # Check for convergence
            break

        y1 = z

    # The second largest eigenvalue
    second_largest_eigenvalue = np.dot(y1, np.dot(a, y1)) / np.dot(y1, y1)
    return second_largest_eigenvalue, y1


if __name__ == '__main__':
    A = np.array([
        [19, 13, 10, 10, 13, -17],
        [13, 13, 10, 10, -11, 13],
        [10, 10, 10, -2, 10, 10],
        [10, 10, -2, 10, 10, 10],
        [13, -11, 10, 10, 13, 13],
        [-17, 13, 10, 10, 13, 19]
    ], dtype="float32") / 12

    # Finding the largest eigenvalue and its eigenvector
    largest_eigenvalue, eigenvector = power_method(A)

    second_largest_eigenval, second_eigenvector = find_second_largest_eigenvalue(A, eigenvector)

    # Print results
    print("Largest Eigenvalue:", largest_eigenvalue)
    print("Corresponding Eigenvector:", eigenvector)
    print("Second Largest Eigenvalue:", second_largest_eigenval)
    print("Corresponding Eigenvector:", second_eigenvector)
