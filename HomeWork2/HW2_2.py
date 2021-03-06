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
    """
    This function asserts if there was a valid return from the eigenvectors and eigenvalues calculation
    :param result:  Return of eigenValuesAndVectors
    :return:        String of assertion's result
    """
    if type(result) == tuple:
        return "Values: \n" + str(result[0]) + '\n Vectors: \n' + str(result[1])
    else:
        return "Can't be displayed because LinAlgError: " + str(result)

def pseudoInv(M):
    """
    This function calculates the Penrose Pseudo-Inverse of a matrix
    :param M:   Matrix to calculate Penrose Pseudo-Inverse
    :return:    Penrose Pseudo-Inverse
    """
    return np.linalg.pinv(M)

if __name__ == '__main__':

    global A
    global B

    # Initialization of matrices A & B
    A = np.array([[1.0, 0.0, 0.0, 0.0, 2.0],
                  [0.0, 0.0, 3.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 4.0, 0.0, 0.0, 0.0]])

    B = np.array([[1.0, 0.0, 0.0,  2.0],
                  [0.0, -1.0, 1.0, 1.0],
                  [1.0, 1.0, -1.0, 0.0],
                  [2.0, 0.0, 0.0,  1.0]])

    # SVD Decomposition of matrix A
    uA, sA, vA = svdDecomp(A)
    print "Single values of matrix A"
    print sA
    print "Matrix A eigen values and vectores, respectively"
    print assertResult(eigenValuesAndVectors(A))
    print "Matrix A Penrose Pseudo-Inverse"
    print pseudoInv(A)
    print "Checking inverse validity A * A+ * A = A"
    print np.allclose(A, np.dot(A, np.dot(pseudoInv(A), A)))

    print "######################"
    uB, sB, vB = svdDecomp(B)
    print "Single values of matrix B"
    print sB
    print "Matrix B eigen values and vectores, respectively"
    print assertResult(eigenValuesAndVectors(B))
    print "Matrix B Penrose Pseudo-Inverse"
    print pseudoInv(B)
    print "Checking inverse validity B * B+ * B = B"
    print np.allclose(B, np.dot(B, np.dot(pseudoInv(B), B)))