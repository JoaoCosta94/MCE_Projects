__author__ = 'JoaoCosta'

import scipy as sp
from scipy.special import *
from numpy import sum
import pylab as pl

def systemSol(o, R, V, E):
    kd = sp.sqrt(E)
    kf = sp.sqrt(abs(V-E))
    return (-kf * jv(o, kd*R) * kvp(o, kf*R)) + (kd * jvp(0, kd*R) * kv(o, kf*R))

def findEnergies(o, R, V0):

    eAttempt = sp.linspace(0.0, V0, 10000)[1:-1]
    sol = systemSol(o, R, V0, eAttempt)

    energies = []
    for i in range(len(sol)-1):
        if (sol[i] * sol[i+1] <= 0):
            energies.append((eAttempt[i] + eAttempt[i+1]) / 2.0)

    return [e for e in energies if e != 0]

def getF(X, Y, R, o, V, E):

    r = sp.sqrt(X**2 + Y**2)

    kd = sp.sqrt(E)
    kf = sp.sqrt(V-E)
    F = jv(o, kd*r) * (r < R) + kv(o, kf*r) * (jv(o, kd*R) / kv(o, kf*R)) * (r >= R)

    return F

def getG(X,Y):
    return sp.cos(sp.arctan2(Y, X))

def normalize(S, spacing):
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