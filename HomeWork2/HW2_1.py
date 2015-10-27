import numpy as np


def invertMatrix(M):
    """
    This function calculates the inverse of a matrix
    :param M:   Matrix to invert
    :return:    Inverse matrix of M
    """
    return np.linalg.inv(M)

def calculateMatrixNorm(M, manual = False):
    """
    This function calculates the norm of a matrix
    :param M:
    :return:    the norm of a matrix
    """
    if (manual):
        col = len(M)
        line = len(M[0])
        sums = []
        for i in range(col):
            sums.append(sum[M[i, :]])
        return max(sums)
    else:
        return np.linalg.norm(M)

def calculateCondNumber(M, iM = None):
    """
    This function calculates the condition number of a given matrix M
    :param M:   Matrix to calculate condition number
    :param iM:  Inverse matrix of m
    :return:    Desired condition number
    """
    if iM:
        # If inverse matrix is passed as an argument, calculates directly
        return calculateMatrixNorm(M) * calculateMatrixNorm(iM)
    else:
        # If inverse matrix is not passed as argument, must first calculate the inverse
        return calculateMatrixNorm(M) * calculateMatrixNorm(invertMatrix(M))


if __name__ == "__main__":
    #Initial Conditions
    e_min = 0.0 # minimum value for e
    e_max = 1.0 # maximum value for e
    nE = 10 # number of values of e desired

    # Arranging the values of e
    # e_array = np.linspace(e_min, e_max)
    e_array = np.array([1.0])

    # Iteration over e to calculate the desired matrices and values
    condNumber_list = []
    for e in e_array:
        # Creation of matrix for respective e value
        M = 0.5 * np.array([[1.0, 1.0 + e], [1.0 + e, 1.0 - e]])
        # Calculation of inverse matrix of M
        iM = invertMatrix(M)
        print iM
        # Calculation of condition number for matrix M
        condNumber_list.append(calculateCondNumber(M, iM))

