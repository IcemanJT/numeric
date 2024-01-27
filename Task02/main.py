import numpy as np
import matplotlib.pyplot as plt

# numpy.linalg.solve() solution:

# [0.1942768  0.1309302  0.14679491 0.16231132 0.09196262 0.13520749
#  0.11957885 0.11199721 0.14035394 0.11669839 0.127685   0.12976704
#  0.11792602 0.12996003 0.12321575 0.12332362 0.12821491 0.12231973
#  0.12616837 0.12550782 0.12357099 0.12637776 0.12428321 0.12490572
#  0.12561555 0.12431504 0.12541532 0.12497054 0.12474606 0.12533124
#  0.12476995 0.12505074 0.12509842 0.12484399 0.1251229  0.12495815
#  0.12496554 0.12507147 0.12493632 0.12502824 0.12500981 0.12496826
#  0.12503211 0.12498276 0.12499873 0.12501358 0.12498422 0.12500987
#  0.12499891 0.12499445 0.12500755 0.12499463 0.1250013  0.12500215
#  0.1249965  0.12500275 0.12499911 0.12499917 0.12500157 0.12499875
#  0.12500033 0.12500048 0.12499928 0.12500033 0.12500033 0.12499928
#  0.12500048 0.12500033 0.12499875 0.12500157 0.12499917 0.12499911
#  0.12500275 0.1249965  0.12500215 0.1250013  0.12499463 0.12500755
#  0.12499445 0.12499891 0.12500987 0.12498422 0.12501358 0.12499873
#  0.12498276 0.12503211 0.12496826 0.12500981 0.12502824 0.12493632
#  0.12507147 0.12496554 0.12495815 0.1251229  0.12484399 0.12509842
#  0.12505074 0.12476995 0.12533124 0.12474606 0.12497054 0.12541532
#  0.12431504 0.12561555 0.12490572 0.12428321 0.12637776 0.12357099
#  0.12550782 0.12616837 0.12231973 0.12821491 0.12332362 0.12321575
#  0.12996003 0.11792602 0.12976704 0.127685   0.11669839 0.14035394
#  0.11199721 0.11957885 0.13520749 0.09196262 0.16231132 0.14679491
#  0.1309302  0.1942768 ]


# ----------- Solving a) ----------- #
def seidel(a, x, b, precision, max_it):
    # dimension of matrix A
    dim = len(a)
    x_new = x.copy()
    xn = 1
    x_norm = []
    it = 0

    while (xn > precision) and (it < max_it):
        # each complete for circle is 1 iteration
        # Gauss Seidel implementation
        for i in range(dim):
            helper = b[i]

            if i < dim - 4:
                helper -= A[i, i+4] * x_new[i+4]
            if i < dim - 1:
                helper -= A[i, i+1] * x_new[i+1]
            if i > 0:
                helper -= A[i, i-1] * x_new[i-1]
            if i > 3:
                helper -= A[i, i-4] * x_new[i-4]

            x_new[i] = helper/A[i][i]

        # calculates how norm changes with each iteration
        xn = np.linalg.norm(x_new - x)
        x_norm.append(xn)
        x = np.copy(x_new)
        it += 1

    return x, x_norm
# ----------- Solving a) ----------- #


# ----------- Solving b) ----------- #
def conjured_gradients(matrix, x1, iterations):
    x0 = np.zeros(len(matrix))
    r0 = e - np.dot(matrix, x0)
    p0 = r0.copy()
    it = 0
    x_norm = []
    while it < iterations:
        ap = np.dot(matrix, p0)
        alpha = np.dot(r0, r0) / np.dot(p0, ap)
        x1 = x0 + alpha * p0
        r1 = r0 - alpha * ap
        x_norm.append(np.linalg.norm(r1))
        if np.linalg.norm(r1) < epsilon:
            break
        beta = np.dot(r1, r1) / np.dot(r0, r0)
        p1 = r1 + beta * p0
        x0, r0, p0 = x1, r1, p1

        it += 1

    return x1, x_norm
# ----------- Solving b) ----------- #


if __name__ == "__main__":

    # ----------- Preps ----------- #
    # Define the size of the matrix
    n = 128

    # Creating matrix A
    A = np.zeros((n, n))

    # i = k, j = m
    for k in range(n):
        for m in range(n):
            if k == m:
                A[k, m] = 4
            elif abs(k - m) == 1:
                A[k, m] = 1
            elif abs(k - m) == 4:
                A[k, m] = 1

    # Creating vector e
    e = np.ones(n)

    # precision
    epsilon = 1e-8

    # ----------- Preps ----------- #

    # ----------- Solving a) ----------- #
    # result for a)
    result_a = np.zeros(n)

    # solves
    result_a, norm_a = seidel(A, result_a, e, epsilon, 42)
    print(result_a)
    # ----------- Solving a) ----------- #

    # ----------- Solving b) ----------- #
    result_b = np.zeros(n)

    result_b, norm_b = conjured_gradients(A, result_b, 42)
    print(result_b)
    # ----------- Solving b) ----------- #

    # plotting
    x_1 = np.arange(0, int(len(norm_a)))
    x_2 = np.arange(0, int(len(norm_b)))
    y1 = norm_a
    y2 = norm_b

    ax = plt.gca()
    ax.set_ylim([0, 0.0001])
    plt.title("Gauss Seidel and Conjugate Gradient comparison")
    plt.xlabel("X: iteration number")
    plt.ylabel("Y: divergence")
    plt.plot(x_1, y1, color="green", label="Gauss-Seidel")
    plt.plot(x_2, y2, color="red", label="Conjugate Gradient")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('my_graph.jpg')
    plt.show()
