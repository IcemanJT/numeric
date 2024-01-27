import numpy as np
np.set_printoptions(suppress=True)


def qr_algorithm(a, num_iterations=50):
    n = a.shape[0]
    eigen_vals = np.zeros(n)

    for _ in range(num_iterations):
        q, r = np.linalg.qr(a)
        a = np.dot(r, q)

    for i in range(n):
        eigen_vals[i] = a[i, i]

    return eigen_vals


def householder_transformation(v):
    v = np.asarray(v, dtype=float)
    n = len(v)

    if n == 0:
        raise ValueError("Input vector must not be empty.")

    # Calculate the norm of the input vector
    norm_v = np.linalg.norm(v)

    if norm_v == 0:
        raise ValueError("Input vector must not be a zero vector.")

    u = v.copy()
    u[0] += np.sign(v[0]) * norm_v
    u = u / np.linalg.norm(u)

    h = np.eye(n) - 2 * np.outer(u, u)

    return h


def matrix_reduction(a):

    n = len(a)
    column = 0

    while column < n-1:
        uncut_vector = a[column, :]
        v = uncut_vector[column + 1:]
        p = householder_transformation(v)
        resized_p = np.eye(n)
        resized_p[column+1:, column+1:] = p
        a = resized_p @ a @ resized_p.T
        column += 1

    return a


if __name__ == '__main__':

    A = np.array([
        [19, 13, 10, 10, 13, -17],
        [13, 13, 10, 10, -11, 13],
        [10, 10, 10, -2, 10, 10],
        [10, 10, -2, 10, 10, 10],
        [13, -11, 10, 10, 13, 13],
        [-17, 13, 10, 10, 13, 19]
    ], dtype="float32") / 12

    print(A)

    A_tri_diag = matrix_reduction(A)
    print("\n ========================================================================\n")

    print(A_tri_diag)

    print("\n ========================================================================\n")

    eigenvalues = qr_algorithm(A_tri_diag)

    print(eigenvalues)

    e_val, e_vec = np.linalg.eig(A)
    print(e_val)

