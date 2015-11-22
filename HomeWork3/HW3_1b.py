__author__ = 'Joao Paulo'

import scipy as sp
import pylab as pl
from scipy.fftpack import fft2, ifft2, fftfreq

def potV(X, Y, V0):
    """
    This function calculates the potential distribution on a given X, Y grid
    :param X:   X points
    :param Y:   y Points
    :param V0:  Value of the potential outside the well
    :return:    Grid with potential distribution
    """
    Z = sp.zeros(X.shape)
    indexes = sp.where((X**2 + Y**2) > R)
    Z[indexes] = V0
    return Z

def Lap(F):
    """
    THis function calculates the laplacian of F using F. Space representation of the operator
    :param F:   phi
    :return:    lap(phi)
    """
    wx = 2.0 * sp.pi * fftfreq(F.shape[0], spacing)
    wy = 2.0 * sp.pi * fftfreq(F.shape[1], spacing)
    fourLap = fft2(F) * (wx**2 + wy**2) * (-1.0)
    return ifft2(fourLap)

def H(F):
    """
    This function calculates H*Phi = (-laplacian + V)*phi
    :param F:   Given phi
    :return:    H*phi
    """
    return V * F - Lap(F)

def invLap(F):
    wx = 2.0 * sp.pi * fftfreq(F.shape[0], spacing)
    wy = 2.0 * sp.pi * fftfreq(F.shape[1], spacing)
    r = fft2(F)/((wx**2 + wy**2) - .5)
    return ifft2(r)

def iH(F, iterations = 10):
    S = sp.zeros(F.shape, complex)
    O = sp.zeros(F.shape, complex)

    O += invLap(F)
    S += O

    for i in range(1, iterations):
        O  = invLap(V * O)
        if i%2 == 0:
            S += O
        else:
            S -= O
    return S

if __name__ == '__main__':
    # Potential well radius
    global R
    global spacing
    global V
    R = 1.0

    # Grid initialization
    nPoints = 2**8
    xyMin = -3.0
    xyMax = 3.0
    xySpace = sp.linspace(xyMin, xyMax, nPoints)
    X, Y = sp.meshgrid(xySpace, xySpace)
    spacing  = X[1]-X[0]

    # Initialization of the potential spacial distribution
    V = potV(X, Y, 1.0)

    # Definition of initial state and normalization
    phi = sp.empty(X.shape, complex)
    phi.real = 2.0 * sp.random.random(X.shape) - 1.0
    phi.imag = 2.0 * sp.random.random(X.shape) - 1.0
    # Normalization
    phi /= (sum((abs(phi)**2)).ravel()*spacing**2)

    de = 1e-14
    hPhi = H(phi)
    e1 = sp.sqrt(sum((abs(hPhi)**2).ravel()) / sum((abs(phi)**2).ravel()))
    e2 = 0.0
    while abs(e2-e1) > de:
        phi = iH(phi)
        phi /= (sum((abs(phi)**2).ravel())*spacing**2)
        e2 = e1
        hPhi = H(phi)
        e2 = sp.sqrt(sum((abs(hPhi)**2).ravel()) / sum((abs(phi)**2).ravel()))

    phi = abs(phi)**2
    levels = sp.linspace(min(phi), max(phi), 1000)
    pl.figure()
    pl.contourf(X, Y, phi, levels = levels)
    pl.colorbar()
    pl.contour(X,Y,V)
    pl.show()