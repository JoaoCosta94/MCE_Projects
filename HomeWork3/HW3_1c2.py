__author__ = 'JoaoCosta'

import scipy as sp
import pylab as pl
from scipy import sum, linalg
import time
import pylab as pl

def potV(X, Y, xyF, a, b, V0):
    Z = sp.zeros(X.shape)
    indexes = sp.where( ( ( (X - 0.5 * xyF) /a )**2 + ((Y - 0.5 * xyF)/ b)**2) > 1.0 )
    Z[indexes] = V0
    return Z

def eigenValues(n, m, xyF):
    return (n*sp.pi/xyF)**2 + (m*sp.pi/xyF)**2

def eigenState(X, Y, n, m, xyF):
    """
    This function calculates the eigen state using a fouriers series
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
    :param nmIndexes:   (n,m) indexes of the states
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

if __name__ == '__main__':

    mPath = 'matrix.npy'

    # Defining grid
    nPoints = 2**8
    xyMax = 3.0
    xySpace = sp.linspace(0.0, xyMax, nPoints)
    X, Y = sp.meshgrid(xySpace, xySpace)
    spacing = xySpace[1] - xySpace[0]

    # Defining problem conditions
    delta = 0.0
    b = 1.0
    a = (1.0 + delta)
    V0 = 100.0

    # Defining state functions max indexes
    N = 15
    M = 15
    # Calculating the indexes of the state functions that will be used
    nm = N * M
    nmIndexes = []
    for n in range(1, N+1):
        for m in range(1, M+1):
            nmIndexes.append((n,m))

    # Defining the potential
    V = potV(X, Y, xyMax, a, b, V0)

    # Creating or loading operator matrix
    # start = time.time()
    # print 'Creating operator matrix. Sit back, this may take a while :)'
    # M = H(X, Y, xyMax, spacing, nm, nmIndexes, V)
    # sp.save(mPath, M)
    # print 'Matrix ready'
    # print 'Took ' + str(time.time() - start) + ' seconds!'
    M = sp.load(mPath)

    # Calculating eigen energies and states
    # values has the energy values
    # weights has the weight for each state on that energy
    # Could be used to represent the state by summing each state with respective weight
    values, weights = linalg.eig(M)
    values = values[values.argsort()]
    weights = weights[values.argsort()]

    state = 0. + 1j*sp.zeros(X.shape)
    for i in range(nm):
       state += weights[i, 0] * eigenState(X, Y, nmIndexes[i][0], nmIndexes[i][1], xyMax)
    state = abs(state)**2
    levels1 = sp.linspace(state.min(), state.max(), 1000)

    pl.figure()
    pl.contourf(X,Y, state, levels = levels1)
    pl.colorbar()
    pl.show()