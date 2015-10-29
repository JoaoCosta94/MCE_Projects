import numpy as np
import pylab as pl

def svdDecomp(M):
    """
    This function calculates the single values of the matrix M
    and the respective matrices U & V
    :param M:   Matrix to perform SVD
    :return:    Matrix U, Single values (S), Matrix V
    """
    return np.linalg.svd(M)

if __name__ == '__main__':

    global A
    global B

    # Initialization of matrices A & B
    A = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 4.0],
                  [0.0, 3.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [2.0, 0.0, 0.0, 0.0]])

    B = np.array([[1.0, 0.0, 1.0,  2.0],
                  [0.0, -1.0, 1.0, 1.0],
                  [1.0, 1.0, -1.0, 0.0],
                  [2.0, 0.0, 0.0,  1.0]])

    uA, sA, vA = svdDecomp(A)
    print sA

    uB, sB, vB = svdDecomp(B)
    print sB