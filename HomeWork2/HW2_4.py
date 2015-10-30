import numpy as np
from scipy.linalg import lu

def luDecomp(M):
    return lu(M)

def extendedMatrix(A, b):
    M = np.zeros((A.shape[0], A.shape[1] + 1), dtype= A.dtype)
    M[:, :-1] = A
    M[:, -1] = b.ravel()
    return M

def gaussElimination(A, b):
    # Cloning original matrices
    M = np.empty_like(A)
    c = np.empty_like(b)
    M[:,:] = A
    c[:] = b
    # U matrix
    n = A.shape[0]
    for k in range(n-1):
        for i in range(k+1, n):
            m = M[i,k] / M[k,k]
            M[i, k] = 0
            for j in range(k+1, n):
                M[i,j] = M[i,j] - m*M[k,j]
            c[i] = c[i] - m*c[k]

    # Solving for x
    x = np.zeros(n)
    k = n-1
    x[k] = c[k]/M[k,k]
    while k >= 0:
        x[k] = (c[k] - np.dot(M[k,k+1:],x[k+1:]))/M[k,k]
        k = k-1
    return x

if __name__ == '__main__':

    # Initialization
    A = np.array([[3.0, 1.0, 1.0],
                  [1.0, 4.0, 2.0],
                  [2.0, 1.0, 5.0]])

    b = np.array([[1.0],
                  [2.0],
                  [-1.0]])

    x0 = np.array([[1.0],
                   [1.0],
                   [1.0]])

    P, L, U = luDecomp(A)

    A_extended = extendedMatrix(A, b)

    print gaussElimination(A, b)