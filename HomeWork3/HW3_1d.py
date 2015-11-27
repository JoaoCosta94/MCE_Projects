__author__ = 'JoaoCosta'

import scipy as sp
from scipy import sum, linalg
import time
import pylab as pl
import platform

def potV(X, Y, xyF, a, b, V0):
    Z = sp.zeros(X.shape)
    indexes = sp.where( ( ( (X - 0.5 * xyF) /a )**2 + ((Y - 0.5 * xyF)/ b)**2) > 1.0 )
    Z[indexes] = V0
    return Z

def eigenValues(n, m, xyF):
    return (n*sp.pi/xyF)**2 + (m*sp.pi/xyF)**2

def eigenState(X, Y, n, m, xyF):
    """
    This function calculates the eigen state using a fourier series
    :param X:       X points
    :param Y:       Y points
    :param n:       n index
    :param m:       m index
    :param xyF:     box limit
    :return:        Eigen state as a fourier series
    """
    return sp.sin((n*sp.pi / xyF) * X) * sp.sin((m*sp.pi / xyF) * Y) * sp.sqrt(4.0 / (xyF**2))

def s1_V_s2(X, Y, xyF, xyS, V, n1, m1, n2, m2):
    """
    This function calculates <Yn1m1|V|Yn2m2>
    :param X:       X points
    :param Y:       Y Points
    :param xyF:     External square box limit
    :param xyS:     Grid points spacing
    :param V:       Potential
    :param n1:      Index n1 of the state 1
    :param m1:      Index m1 of the state 1
    :param n2:      Index n2 of the state 2
    :param m2:      Index m2 of the state 2
    :return:
    """

    state1 = eigenState(X, Y, n1, m1, xyF)
    state2 = eigenState(X, Y, n2, m2, xyF)
    return sum(state1 * V * state2) * xyS**2

def H(X, Y, xyF, xyS, nm, nmIndexes, V):
    """
    This function creates a matrix of the equivalent application of a potential to a free particle,
    like a perturbation theory calculating the 'overlap' of N & M states of X & Y directions
    :param xyF:         External square box limit
    :param xyS:         Grid points spacing
    :param nm:          Total of states used
    :param nmIndexes:   (n,m) indexes of the state basis
    :param V:           Potential
    :return:            Returns the representation of the H operator with given V
    """

    hMatrix = sp.zeros((nm, nm), dtype = float)

    # Matrix elements
    for i in range(nm-1):
        for j in range(i+1, nm):
            # Using closest neighbours states -> Perturbation of V
            hMatrix[i][j] += s1_V_s2(X, Y, xyF, xyS, V, nmIndexes[i][0], nmIndexes[i][1], nmIndexes[j][0], nmIndexes[j][1])
            hMatrix[j][i] += hMatrix[i][j]

    # Matrix diagonal elements
    for i in range(nm):
        # Using self state -> Average of V
        hMatrix[i][i] = eigenValues(nmIndexes[i][0], nmIndexes[i][1], xyF)
        hMatrix[i][i] += s1_V_s2(X, Y, xyF, xyS, V, nmIndexes[i][0], nmIndexes[i][1], nmIndexes[i][0], nmIndexes[i][1])

    return hMatrix

def calculateState(o, nmIndexes, weights):
    """
    This functions calculates the o'th bound state
    :param o:               order of the bound state
    :param nmIndexes:       (n,m) indexes of the state basis
    :param weights:         coefficients for each state function
    :return:                returns the square of the module of the eigen state
    """
    state = 0. + 1j*sp.zeros(X.shape)
    for i in range(nm):
       state += weights[i, o] * eigenState(X, Y, nmIndexes[i][0], nmIndexes[i][1], xyMax)
    state = abs(state)**2
    return state


if __name__ == '__main__':

    # Some parameters maybe be changed but if so, please delete all the matrices in the matrices folder and recalculate them
    # These parameters are flagged.

    # Defining grid
    # Don't change please!
    nPoints = 2**8
    xyMax = 8.0
    xySpace = sp.linspace(0.0, xyMax, nPoints)
    X, Y = sp.meshgrid(xySpace, xySpace)
    spacing = xySpace[1] - xySpace[0]

    # Defining problem conditions
    # Don't change these or matrices will have to be calculated again! (It works but you'll have to wait)
    V0 = 100.0
    deltaArray = sp.arange(0.0, 1.1, 0.1)
    b = 1.0

    # Defining state functions max indexes
    # Don't change this please!!!
    N = 15
    M = 15
    # Calculating the indexes of the state functions that will be used
    nm = N * M
    nmIndexes = []
    for n in range(1, N+1):
        for m in range(1, M+1):
            nmIndexes.append((n,m))

    # Calculations begin here
    e0Array = []
    e1Array = []
    e2Array = []
    e3Array = []
    e4Array = []
    for delta in deltaArray:
        a = (1.0 + delta)

        # Defining the potential
        V = potV(X, Y, xyMax, a, b, V0)

        # Creating or loading operator matrix
        if (platform.system() == 'Windows'):
            mPath = 'Operator_Matrices\\delta_' + str(delta) + 'V0_' + str(V0) + '.npy'
        else:
            mPath = 'Operator_Matrices/delta_' + str(delta) + 'V0_' + str(V0) + '.npy'

        try:
            M = sp.load(mPath)
            print 'Matrix will be loaded'
        except:
            start = time.time()
            print 'Creating operator matrix. Sit back, this may take a while :)'
            M = H(X, Y, xyMax, spacing, nm, nmIndexes, V)
            sp.save(mPath, M)
            print 'Matrix ready'
            print 'Took ' + str(time.time() - start) + ' seconds!'

        # Calculating eigen energies
        values = linalg.eig(M)[0]
        values = values[values.argsort()]
        e0Array.append(values[0].real)
        e1Array.append(values[1].real)
        e2Array.append(values[2].real)
        e3Array.append(values[3].real)
        e4Array.append(values[4].real)

    e0Array = sp.array(e0Array)
    e1Array = sp.array(e1Array)
    e2Array = sp.array(e2Array)
    e3Array = sp.array(e3Array)
    e4Array = sp.array(e4Array)

    pl.figure()
    pl.plot(deltaArray, e0Array)
    pl.plot(deltaArray, e1Array)
    pl.plot(deltaArray, e2Array)
    pl.plot(deltaArray, e3Array)
    pl.plot(deltaArray, e4Array)

    pl.show()