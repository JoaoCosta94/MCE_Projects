__author__ = 'JoaoCosta'

import scipy as sp
from scipy.special import *
from numpy import sum
import pylab as pl

def systemSol(o, R, V, E):
    """
    Calculates the determinant of the boundary conditions problem system
    :param o:   Order of the solution
    :param R:   Radius of the potential well
    :param V:   Potential well depth
    :param E:   Possible energy solution
    :return:    Determinant of the boundary conditions problem system
    """
    kd = sp.sqrt(E)
    kf = sp.sqrt(abs(V-E))
    return (-kf * jv(o, kd*R) * kvp(o, kf*R)) + (kd * jvp(0, kd*R) * kv(o, kf*R))

def findEnergies(o, R, V0):
    """
    This function calculates the energies of the bound states
    :param o:   O-th energy (order)
    :param R:   Potential well Radius
    :param V0:  Potential well depth
    :return:
    """
    eAttempt = sp.linspace(0.0, V0, 10000)[1:-1]
    sol = systemSol(o, R, V0, eAttempt)

    energies = []
    for i in range(len(sol)-1):
        if (sol[i] * sol[i+1] <= 0):
            energies.append((eAttempt[i] + eAttempt[i+1]) / 2.0)

    return [e for e in energies if e != 0]

def getF(X, Y, R, o, V, E):
    """
    This function calculates the radial dependency of phi given the energy solutions
    :param X:   X points
    :param Y:   Y Points
    :param R:   Potential well radius
    :param o:   Solution order
    :param V:   Potential values
    :param E:   Energy value for given order
    :return:    F(r)
    """

    r = sp.sqrt(X**2 + Y**2)

    kd = sp.sqrt(E)
    kf = sp.sqrt(V-E)
    F = jv(o, kd*r) * (r < R) + kv(o, kf*r) * (jv(o, kd*R) / kv(o, kf*R)) * (r >= R)

    return F

def getG(X,Y):
    """
    This function calculates the angular components of a state
    :param X:   X Points
    :param Y:   Y Points
    :return:    Arctan(Y/X)
    """
    return sp.cos(sp.arctan2(Y, X))

def normalize(S, spacing):
    """
    This function calculates the normalization of |phi|^2
    :param S:           State phi
    :param spacing:     (X,Y) grid spacing between points
    :return:            Normalized |phi|^2 state
    """
    A = sum(S**2) * spacing ** 2
    return abs(S) ** 2 / A

if __name__ == '__main__':
    # Problem characteristics
    R = 1.0
    V0 = 115.0

    E0 = findEnergies(0, R, V0)[0]
    E1 = findEnergies(1, R, V0)[0]

    # Grid Parameters
    nPoints = 2**8
    xyMin = -4.0
    xyMax = 4.0
    xySpace = sp.linspace(xyMin, xyMax, nPoints)
    X, Y = sp.meshgrid(xySpace, xySpace)
    spacing = xySpace[1]-xySpace[0]


    # Eigen States and plotting
    phi1 = normalize(getF(X, Y, R, 0, V0, E0), spacing)
    levels1 = sp.linspace(phi1.min(), phi1.max(), 1000)

    phi2 = normalize(getF(X, Y, R, 1, V0, E1) * getG(X,Y), spacing)
    levels2 = sp.linspace(phi2.min(), phi2.max(), 1000)

    CIRC = 1.0 * ((X**2 + Y**2) < R)

    pl.figure("First State")
    pl.contourf(X, Y, phi1, levels = levels1)
    pl.colorbar()
    pl.contour(X, Y, CIRC)

    pl.figure("Second State")
    pl.contourf(X, Y, phi2, levels = levels2)
    pl.colorbar()
    pl.contour(X, Y, CIRC)

    pl.show()