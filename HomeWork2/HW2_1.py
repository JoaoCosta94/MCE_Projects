import numpy as np
import pylab as pl

def analyticalInverse(e):
    """
    This function returns the inverse matrix of
    [[1.0,     1.0],
    [1.0+e, 1.0-e]]
    :param e:   Matrix variables
    :return:    Inverse matrix
    """
    return np.array([[e - 1.0, 1.0], [e + 1.0, -1.0]], dtype =np.float32) / e

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
    Manual mode iterates each column and sums it's elements and return the maximum sum
    The other mode uses numpy.linalg.norm to calculate the norm of M
    :param M:   Matrix to calculate the norm of
    :return:    the norm of a matrix
    """
    if (manual):
        col = len(M)
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
    if iM == None:
        # If inverse matrix is passed as an argument, calculates directly
        return calculateMatrixNorm(M) * calculateMatrixNorm(invertMatrix(M))
    else:
        # If inverse matrix is not passed as argument, must first calculate the inverse
        return calculateMatrixNorm(M) * calculateMatrixNorm(iM)


if __name__ == "__main__":
    #Initial Conditions
    e_min = 10.0**(-10) # minimum value for e
    e_max = 10.0 # maximum value for e
    nE = 1000 # number of values of e desired

    # Arranging the values of e
    e_array = np.linspace(e_min, e_max, nE)
    # e_array = np.array([1.0])
    # print e_array

    # Iteration over e to calculate the desired matrices and values
    condNumber_list_numerical = []
    condNumber_list_analytical = []
    for e in e_array:

        # Creation of matrix for respective e value
        M = 0.5 * np.array([[1.0, 1.0], [1.0 + e, 1.0 - e]])
        # Calculation of inverse matrix of M
        iM = invertMatrix(M)
        # Calculation of condition number for matrix M
        condNumber_list_numerical.append(calculateCondNumber(M, iM))

        # Calculations with analytical inverse matrix
        iM = analyticalInverse(e)
        condNumber_list_analytical.append(calculateCondNumber(M, iM))

    print '(Numerical) Condition number for e = ' + str(e_array[0])
    print condNumber_list_numerical[0]

    print '(Analytical) Condition number for e = ' + str(e_array[0])
    print condNumber_list_analytical[0]

    print 'Analytical Inverse'
    print analyticalInverse(1.0E-10)

    print 'Numerical Inverse'
    print invertMatrix(0.5 * np.array([[1.0, 1.0], [1.0 + 1.0E-10, 1.0 - 1.0E-10]]))

    pl.figure("Condition number")
    # TODO: Change epsilon hardcoded to raw string
    pl.title(r'Condition number($\epsilon$)')
    pl.plot(e_array, condNumber_list_numerical, label = 'N. inverse')
    pl.plot(e_array, condNumber_list_analytical, label = 'A. inverse')
    pl.yscale('log')
    pl.ylabel('Log(C. Number)')
    pl.xlabel(r'$\epsilon$')
    pl.legend()

    pl.show()


