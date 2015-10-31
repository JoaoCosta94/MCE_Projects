import numpy as np
from scipy.linalg import lu, solve
import pylab as pl

def luDecomp(M):
    return np.tril(M), np.triu(M)

def getDiagonal(M):
    return np.diag(np.diag(M))


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

def Seidel(A, b, x0, iterations):
    # Obtaining M = D + L an N = U matrices
    L, U = luDecomp(A)
    U = A - L
    # Initializing with given seed / guess
    x = np.empty_like(x0)
    x[:] = x0[:]
    convergence = [abs(x - analyticalSol)]
    # Iteration
    for i in range(iterations):
        aux = - np.dot(U, x) + b
        x = np.dot(np.linalg.inv(L), b - np.dot(U, x))
        convergence.append(abs(x - analyticalSol))
    return x, np.array(convergence)

def SORelaxation(A, b, x0, l, iterations):
    L, U = luDecomp(A)
    D = getDiagonal(A)
    L = L - D
    U = U -D
    return "cenas"

if __name__ == '__main__':

    global analyticalSol

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

    iterations = 10 # Number of iterations for Gauss-Seidel method

    # analyticalSol = np.array([])
    analyticalSol = solve(A, b.ravel())
    print "Analytical Solution"
    print analyticalSol

    # Solving with Ax = b different methods
    # By Gauss elimination
    gauss = gaussElimination(A, b.ravel())

    # By Gauss-Seidel iterative method
    seidel, seidelConv = Seidel(A, b.ravel(), x0.ravel(), iterations)
    # Convergence on each axis for this method
    sXConv = seidelConv[:,0]
    sYConv = seidelConv[:,1]
    sZConv = seidelConv[:,2]

    # By successive over relaxation
    print SORelaxation(A,b,x0,1,1)

    print "Gauss Elimination Solution"
    print gauss
    print "Gauss-Seidel Solution"
    print seidel


    # Plotting convergence
    # TODO: FIx Scales
    itArray = np.arange(0, iterations+1, 1)

    pl.figure("Gauss-Seidel Method")
    pl.subplot(311)
    pl.title("Convergence of x1")
    pl.xlabel("Iteration")
    pl.ylabel("Convergence")
    pl.plot(itArray, sXConv)

    pl.subplot(312)
    pl.title("Convergence of x2")
    pl.xlabel("Iteration")
    pl.ylabel("Convergence")
    pl.plot(itArray, sYConv)

    pl.subplot(313)
    pl.title("Convergence of x3")
    pl.xlabel("Iteration")
    pl.ylabel("Convergence")
    pl.plot(itArray, sZConv)

    pl.subplots_adjust(hspace = 0.7)

    pl.show()