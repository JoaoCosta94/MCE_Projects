__author__ = 'Joao Paulo'

import scipy as sp
import pylab as pl
from scipy.fftpack import fft2, ifft2, fftfreq
from scipy import linalg
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
    Z[indexes] = V0 - dV
    return Z

def Lap(F, wx, wy):
    """
    THis function calculates the laplacian of F using F. Space representation of the operator
    :param F:   phi
    :return:    lap(phi)
    """
    wx, wy = sp.meshgrid(2.0 * sp.pi * fftfreq(F.shape[0], spacing), 2.0 * sp.pi * fftfreq(F.shape[1], spacing))
    fourLap = fft2(F) * (wx**2 + wy**2)
    return ifft2(fourLap)

def H(F, wx, wy):
    """
    This function calculates H*Phi = (-laplacian + V)*phi
    :param F:   Given phi
    :return:    H*phi
    """
    return V * F - Lap(F, wx, wy)

def invLap(F):
    """
    This function  calculates the inverse laplacian
    :param F:   Given function values
    :return:    lap^-1(F)
    """
    wx, wy = sp.meshgrid(2.0 * sp.pi * fftfreq(F.shape[0], spacing), 2.0 * sp.pi * fftfreq(F.shape[1], spacing))
    r = fft2(F)/((wx**2 + wy**2)) * (-1.0)
    return ifft2(r)

# def iH(F, iterations = 10):
#     A = sp.zeros(F.shape, complex)
#     B = sp.zeros(F.shape, complex)
#
#     B -= invLap(F)
#     A += B
#
#     for i in range(1, iterations):
#         B  = -invLap(V * B)
#         if i%2 == 0:
#             A += B
#         else:
#             A -= B
#     return A

def firstState(X, Y):
    """
    This function calculates the first state distribution according to the given  potential
    :param X:   X axis points
    :param Y:   y axis points
    :return:    |phi|^2
    """

    # Definition of initial state and normalization
    phi = 2.0*(sp.random.random(X.shape)-0.5) + 1j * 2.0*(sp.random.random(X.shape)-0.5)
    # Frequencies
    wx, wy = sp.meshgrid(2.0 * sp.pi * fftfreq(phi.shape[0], spacing), 2.0 * sp.pi * fftfreq(phi.shape[1], spacing))

    # Normalization
    phi /= sum(abs(phi)**2)*spacing**2

    energyTreshold = 1e-14
    IK2 =  1.0 / (-(wx**2 + wy**2) + 1.0)
    eO = 1.0
    eN = 0.0
    while abs(eN-eO) > energyTreshold:
        phi = ipm(phi, IK2)
        phi /= sum(abs(phi)**2)*spacing**2
        eO = eN
        hPhi = H(phi, wx, wy)
        eN = sp.sqrt(sum(abs(hPhi)**2) / sum(abs(phi)**2))

    phi = abs(phi)**2
    return phi, eO

if __name__ == '__main__':
    # Global Parameters
    global R
    global spacing
    global V
    global dV
    global a
    global b
    R = 1.0
    dV = 1e-4
    b = 1.0
    delta = 0.5
    a = (1.0 + delta)

    # Grid initialization
    nPoints = 2**8
    xyMin = -3.0
    xyMax = 3.0
    xySpace = sp.linspace(xyMin, xyMax, nPoints)
    X, Y = sp.meshgrid(xySpace, xySpace)
    spacing  = xySpace[1]-xySpace[0]

    # Initialization of the potential spacial distribution
    V = potV(X, Y, 1.0)

    phi1, e1 = firstState(X,Y)

    levels = sp.linspace(phi1.min(), phi1.max(), 1000)
    pl.figure()
    pl.contourf(X, Y, phi1, levels = levels)
    pl.colorbar()
    pl.contour(X,Y,V)
    pl.show()