import numpy as np
np.set_printoptions(suppress=True, linewidth=150)


if __name__ == '__main__':
    H = np.array([[0, 1, 0, -1j],
                  [1, 0, -1j, 0],
                  [0, 1j, 0, 1],
                  [1j, 0, 1, 0]])

    A = np.array([[0, 1, 0, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]], dtype=complex)

    B = np.array([[0, 0, 0, -1],
                  [0, 0, -1, 0],
                  [0, 1, 0, 0],
                  [1, 0, 0, 0]], dtype=complex)

    x = np.random.rand(len(H))
    y = np.random.rand(len(H))

    # noinspection PyTypeChecker
    H_1 = np.block([[A, -1 * B], [B, A]])
    print(H_1)

    eigen_vals, eigen_vecs = np.linalg.eigh(H_1)
    print("\n====================\n")
    print(eigen_vals)
    print(eigen_vecs)
    print("\n====================\n")



    print("\n====================\n")

    for vector in eigen_vecs[::2]:
        x = vector[:len(H)]
        y = vector[len(H):]  * -1j
        new_vector = x + y
        print(np.linalg.norm(new_vector))

    print("\n====================\n")



    print("\n====================\n")
    e_vals, e_vecs = np.linalg.eigh(H)
    print(e_vals)
    print(e_vecs)
    print("\n====================\n")



