import numpy as np
from scipy.linalg import lu, solve
import pylab as pl
import time

def luDecomp(M):
    """
    This function calculates the Lower (L) and Upper (U) triangular matrices of M
    :param M:   Matrix M
    :return:    Matrices L and U, respectively
    """
    return np.tril(M), np.triu(M)

def getDiagonal(M):
    """
    This function obtains the main diagonal of a matrix
    :param M:   Matrix to obtain diagonal
    :return:    Matrix with the diagonal elements of M
    """
    return np.diag(np.diag(M))


def extendedMatrix(A, b):
    """
    Given a matrix A and a vector b this function extends A with the values of b on a new column
    :param A:   Matrix A
    :param b:   Vector b
    :return:    Extended matrix
    """
    M = np.zeros((A.shape[0], A.shape[1] + 1), dtype= A.dtype)
    M[:, :-1] = A # All values except last column are equal to those of A
    M[:, -1] = b.ravel() # The values of last column are to the values of vector b
    return M

def gaussElimination(A, b):
    """
    This function calculates the solution of Ax=b using Gauss elimination process
    :param A:               Matrix A
    :param b:               Vector b
    :return:                Obtained x solution
    """
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
    """
    This function approximates a numerical solution of Ax=b using Gauss-Seidel method
    :param A:               Matrix A
    :param b:               Vector b
    :param x0:              Initial seed / guess for the solution
    :param iterations:      Number of iterations
    :return:                Approximated x solution
    """
    # Obtaining M = D + L an N = U matrices
    L, U = luDecomp(A)
    U = A - L
    # Initializing with given seed / guess
    x = np.empty_like(x0)
    x[:] = x0[:]
    convergence = [np.average(abs(x - analyticalSol))]
    # Iteration
    for i in range(iterations):
        aux = - np.dot(U, x) + b
        x = np.dot(np.linalg.inv(L), aux)
        convergence.append(np.average(abs(x - analyticalSol)))
    return x, np.array(convergence)

def SORelaxation(A, b, x0, l, iterations):
    """
    This function approximates a numerical solution of Ax=b using successive relaxation method
    :param A:               Matrix A
    :param b:               Vector b
    :param x0:              Initial seed / guess for the solution
    :param l:               lambda value for convergence speed up
    :param iterations:      Number of iterations
    :return:                Approximated x solution
    """
    # Obtaining D L U matrices
    L, U = luDecomp(A)
    D = getDiagonal(A)
    L = L - D
    U = U - D
    # Auxiliary matrices corresponding to the ones need for the operations
    Mi = np.linalg.inv(D/l + L) # Inverse of matrix M
    N  = U - (1.0-l)*D/l
    # Initializing with given seed / guess
    x = np.empty_like(x0)
    x[:] = x0[:]
    convergence = [np.average(abs(x - analyticalSol))]
    for i in range(iterations):
        aux = b - np.dot(N, x)
        x = np.dot(Mi, aux)
        convergence.append(np.average(abs(x - analyticalSol)))
    return x, np.array(convergence)

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
    lambda_array = np.array([0.0, 0.02, 0.5, 0.7, 0.8, 1.0, 1.5, 1.7]) # Lambda values for successive over relaxation method
    # lambda_array = np.array([1.0])
    l_default = 0.8 # Default lambda for

    # analyticalSol = np.array([])
    analyticalSol = solve(A, b.ravel())
    print "Analytical Solution"
    print analyticalSol

    # Solving with Ax = b different methods
    # By Gauss elimination
    start = time.time()
    gauss = gaussElimination(A, b.ravel())
    print "Gauss Elimination Time"
    print time.time() - start

    # By Gauss-Seidel iterative method
    start = time.time()
    seidel, seidelConv = Seidel(A, b.ravel(), x0.ravel(), iterations)
    print "Gauss-Seidel Elimination Time"
    print time.time()-start

    # By successive over relaxation using different values for lambda defined above
    sorConv_list = []
    for l in lambda_array:
        sor, sorConv = SORelaxation(A,b.ravel(),x0.ravel(), l, iterations)
        sorConv_list.append(sorConv)
    sorConv_list = np.array(sorConv_list)

    start = time.time()
    sor, sorConv = SORelaxation(A,b.ravel(),x0.ravel(), l_default, iterations)
    print "Successive over relaxation Time with l = " + str(l_default)
    print time.time()-start

    print "Gauss Elimination Solution"
    print gauss
    print "Gauss-Seidel Solution"
    print seidel
    print "Gauss-Seidel Average Error"
    print np.average(abs(seidel-analyticalSol)/analyticalSol) * 100

    print "Successive over relaxation Solution with l = " + str(l_default)
    print sor
    print "Successive over relaxation Average Error with l = " + str(l_default)
    print np.average(abs(sor-analyticalSol)/ analyticalSol) * 100

    ##############################################################################################

    # Plotting convergence
    itArray = np.arange(0, iterations+1, 1)

    # Plotting of convergence for Guass-Seidel method
    pl.figure("Gauss-Seidel Convergence")
    pl.title("Gauss-Seidel Convergence")
    pl.xlabel("iterations")
    pl.ylabel("average error")
    pl.plot(itArray, seidelConv)

    # Plotting of convergence for 'Successive Over Relaxation' method
    pl.figure("Successive Over Relaxation Convergence")
    pl.title("Successive Over Relaxation Convergence")
    pl.xlabel("iterations")
    pl.ylabel("average error")
    for i in range(len(lambda_array)):
        pl.plot(itArray, sorConv_list[i], label = r'$\lambda$ = ' + str(lambda_array[i]))
    pl.legend()

    pl.show()