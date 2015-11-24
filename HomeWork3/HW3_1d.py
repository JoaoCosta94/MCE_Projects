__author__ = 'Joao Paulo'

import scipy as sp
import pylab as pl
from scipy.fftpack import fft2, ifft2, fftfreq
from numpy import sum

def potV(X, Y, V0):
    """
    This function calculates the potential distribution on a given X, Y grid
    :param X:   X points
    :param Y:   y Points
    :param V0:  Value of the potential outside the well
    :return:    Grid with potential distribution
    """
    Z = sp.zeros(X.shape)
    indexes = sp.where(((X/a)**2 + (Y/b)**2) > R)
    Z[indexes] = V0
    return Z

def Lap(F):
    """
    THis function calculates the laplacian of F using F. Space representation of the operator
    :param F:   phi
    :return:    lap(phi)
    """
    wx, wy = sp.meshgrid(2.0 * sp.pi * fftfreq(F.shape[0], spacing), 2.0 * sp.pi * fftfreq(F.shape[1], spacing))
    fourLap = fft2(F) * (wx**2 + wy**2) * (-1.0)
    return ifft2(fourLap)

def H(F, V):
    """
    This function calculates H*Phi = (-laplacian + V)*phi
    :param F:   Given phi
    :return:    H*phi
    """
    return V * F - Lap(F)

def invLap(F):
    """
    This function  calculates the inverse laplacian
    :param F:   Given function values
    :return:    lap^-1(F)
    """
    wx, wy = sp.meshgrid(2.0 * sp.pi * fftfreq(F.shape[0], spacing), 2.0 * sp.pi * fftfreq(F.shape[1], spacing))
    r = fft2(F)/((wx**2 + wy**2) - 0.01) * (-1.0)
    return ifft2(r)

def iH(F, V, iterations = 10):
    S = sp.zeros(F.shape, complex)
    O = sp.zeros(F.shape, complex)

    O -= invLap(F)
    S += O

    for i in range(1, iterations):
        O  = -invLap(V * O)
        if i%2 == 0:
            S += O
        else:
            S -= O
    return S

def firstState(X, Y, V):
    """
    This function calculates the first state distribution according to the given  potential
    :param X:   X axis points
    :param Y:   y axis points
    :return:    |phi|^2
    """

    # Definition of initial state and normalization
    phi = sp.empty(X.shape, complex)
    phi.real = 2.0 * sp.random.random(X.shape) - 1.0
    phi.imag = 2.0 * sp.random.random(X.shape) - 1.0
    # Normalization
    phi /= sum(abs(phi)**2)*spacing**2

    energyThreshold = 1e-14
    hPhi = H(phi, V)
    e1 = sp.sqrt(sum(abs(hPhi)**2)) / sp.sqrt(sum(abs(phi)**2))
    e2 = 0.0
    while abs(e2-e1) > energyThreshold:
        phi = iH(phi, V)
        phi /= sum(abs(phi)**2)*spacing**2
        e1 = e2
        hPhi = H(phi, V)
        e2 = sp.sqrt(sum(abs(hPhi)**2)) / sp.sqrt(sum(abs(phi)**2))

    phi = abs(phi)**2
    return (phi, e1)

if __name__ == '__main__':
    # Global Parameters
    global R
    global spacing
    global a
    global b
    R = 1.0
    b = 1.0

    # Grid initialization
    nPoints = 2**8
    xyMin = -3.0
    xyMax = 3.0
    xySpace = sp.linspace(xyMin, xyMax, nPoints)
    X, Y = sp.meshgrid(xySpace, xySpace)
    spacing  = xySpace[1]-xySpace[0]

    # Delta Parameters
    di = 0.0
    df = 1.0
    nDelta = 20
    deltaArray = sp.linspace(di, df, nDelta)

    e1Array = []
    for delta in deltaArray:
        # Initialization of the potential spacial distribution
        a = 1.0 + delta
        V = potV(X, Y, 1.0)

        e1Array.append(firstState(X,Y,V)[1])

    eArray = sp.array(e1Array)

    pl.figure()
    pl.title('1st Energy vs.  ' + r'$\delta$')
    pl.plot(deltaArray, e1Array)
    pl.xlabel(r'$\delta$')
    pl.ylabel('1st energy')
    pl.show()