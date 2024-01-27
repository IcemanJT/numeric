import numpy as np
import scipy.linalg as la

def search_eigenvec(p, l, u, v0, iterations, precision):
    for _ in range(iterations):
        z = la.solve_triangular(l, v0[p, :], lower=True)
        eigen_vector = la.solve_triangular(u, z)
        eigen_vector /= np.linalg.norm(eigen_vector)
    
        if (np.linalg.norm(np.abs(v0) - np.abs(eigen_vector)) < precision or
            np.allclose(np.abs(v0), np.abs(eigen_vector), atol=precision)):
            break
    
    return eigen_vector
    



if __name__ == "__main__":
    
    A = np.array([[2, -1, 0, 0, 1],
                        [-1, 2, 1, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 1, 2, -1],
                        [1, 0, 0, -1, 2]])
       
    n = A.shape[0] 
                
    eigen_value = 0.38197
    I = np.identity(n)
    B = A - eigen_value * I
    
    print(B)

    P, L, U = la.lu(B, p_indices=True)
    
    print(L)
    print(U.flatten())
    
    v = np.random.random(n).reshape(n, 1)
    v /= np.linalg.norm(v)

    result = search_eigenvec(P, L, U, v, 1000, 1e-10)
    result = result.flatten()
    
    print(result)

