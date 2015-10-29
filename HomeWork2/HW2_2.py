import numpy as np

def svdDecomp(M):
    """
    This function calculates the single values of the matrix M
    and the respective matrices U & V
    :param M:   Matrix to perform SVD
    :return:    Matrix U, Single values (S), Matrix V
    """
    return np.linalg.svd(M)

def eigenValuesAndVectors(M):
    """
    This function calculates the eigen values and vectors of a matrix, if possible
    :param M:   Matrix to calculate eigen values and vectors
    :return:    Matrix's M eigen values and vectors if they are defined, or error
    """
    try:
        return np.linalg.eig(M)
    except np.linalg.linalg.LinAlgError as error:
        return error

def assertResult(result):
    if type(result) == tuple:
        return "Values: \n" + str(result[0]) + '\n Vectors: \n' + str(result[1])
    else:
        return "Can't be displayed because LinAlgError: " + str(result)

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

    # SVD Decomposition of matrix A
    uA, sA, vA = svdDecomp(A)
    print "Single values of matrix A"
    print sA
    print "Matrix A eigen values and vectores, respectively"
    print assertResult(eigenValuesAndVectors(A))

    print "######################"
    uB, sB, vB = svdDecomp(B)
    print "Single values of matrix B"
    print sB
    print "Matrix B eigen values and vectores, respectively"
    print assertResult(eigenValuesAndVectors(B))