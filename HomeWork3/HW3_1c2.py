__author__ = 'JoaoCosta'

import scipy as sp
import pylab as pl

def potV(X, Y, xyF, a, b, V0):
    Z = sp.zeros(X.shape)
    indexes = sp.where( ( ( (X - 0.5 * xyF) /a )**2 + ((Y - 0.5 * xyF)/ b)**2) > 1.0 )
    Z[indexes] = V0
    return Z

def Pertubation_V_State(X, Y, xyF, xyS, n1, m1, n2, m2):
    return sum(eigFuncD0(X, Y, n1, m1, xyF) * V * eigFuncD0(X, Y, n2, m2, xyF)) * xyS**2

def D(xyF, xyS, N, M, a, b, V):

    xySpace = sp.linspace(0,0, xyF, xyF/xyS)
    X, Y = sp.meshgrid(xySpace, xySpace)
    nm = N * M
    nmIndexes = []
    for n in range(1, N+1):
        for m in range(1, M+1):
            nmIndexes.append((n,m))

    dMatrix = sp.zeros((nm,nm), dtype = float)

    for i in range(nm-1):
        for j in range(i+1, nm):
            dMatrix[i][j] += Pertubation_V_State(X, Y, xyF, xyS, nmIndexes[i][0], nmIndexes[i][1], nmIndexes[j][0], nmIndexes[j][1])
            dMatrix[j][i] += dMatrix[i][j]

if __name__ == '__main__':

    # Defining grid
    nPoints = 2**8
    xyMin = -3.0
    xyMax = 3.0
    xySpace = sp.linspace(xyMin, xyMax, nPoints)
    X, Y = sp.meshgrid(xySpace, xySpace)
    spacing = xySpace[1] - xySpace[0]

    # Defining problem conditions
    delta = 0.5
    b = 1.0
    a = (1.0 + delta)
    V0 = 100.0

    # Defining the potential
    V = potV(X, Y, a, b, V0)

    # Creating operator matrix
