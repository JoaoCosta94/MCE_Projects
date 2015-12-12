__author__ = 'JoaoCosta'

import scipy as sp
from scipy.sparse import linalg
from scipy import sparse
from scipy import sum
import pylab as pl
from time import time

def potential_well(X, Y, x0, y0, x1, R, v0):
    """
    This function generates the potential well
    """
    V = v0 * ((X-x0)**2 + (Y-y0)**2 > R**2) * ((X-x1)**2 + (Y-y0)**2 > R**2)
    return V

def absorving_borders_box(X, Y, xyT, xyMax, vM):
    """
    This function generates the absorbing potential on the borders
    """
    border = sp.zeros(X.shape, dtype = complex)
    idx = sp.where(abs(X) > (xyMax - xyT))
    idy = sp.where(abs(Y) > (xyMax - xyT))
    border[idx] += vM * ((abs(X[idx]) - xyMax + xyT)**2 * 1j - (abs(X[idx]) - xyMax + xyT)**2)
    border[idy] += vM * ((abs(Y[idy]) - xyMax + xyT)**2 * 1j - (abs(Y[idy]) - xyMax + xyT)**2)
    return border

def lap(shape, spacing):
    """
    This function generates the laplacian operator
    """
    n = shape[0]*shape[1]
    L = -4.*sparse.eye(n, n, 0) + sparse.eye(n, n, 1) + sparse.eye(n, n, -1) + sparse.eye(n, n, shape[1]) + sparse.eye(n, n, -shape[1])
    return L / spacing**2

def initial_state(x0, y0, X, Y):
    """
    This function generates the initial state with given parameters
    """
    delta = 0.1 #(R-x0) / 20.0
    psi = sp.exp(-((X-x0)**2 + (Y-y0)**2) / delta**2)
    return psi

def normalize(state, spacing):
    """
    This function normalizes a given state
    """
    N = sp.sqrt(sum(abs(state)**2) * spacing**2)
    return state / N

def split_step_fourier(state, V, spacing, dt):
    """
    This function evolves the state by a time step using the split step Fourier method
    """
    nX = state.shape[0]
    nY = state.shape[1]
    Wx , Wy = sp.meshgrid(2.0 * sp.pi * pl.fftfreq(nX, spacing), 2.0 * sp.pi * pl.fftfreq(nY, spacing))

    stateNew = sp.exp(-1j * dt * V) * state
    stateNew = pl.fft2(stateNew)
    stateNew = pl.exp(-1j * dt * (Wx**2 + Wy**2)) * stateNew

    return pl.ifft2(stateNew)

def hamiltonian_operator(X, Y, spacing, xyT, xyMax, x0, y0, R, v0, vM):
    """
    This function generates the Hamiltonian Operator matrix with given potential and absorbing borders box
    """
    L = lap(X.shape, spacing)
    V = potential_well(X, Y, x0, y0, R, v0) + absorving_borders_box(X, Y, xyT, xyMax, vM)
    return -L + sparse.diags(V.ravel(),0, format = 'dia')

def theta_family_step(F, u, theta, dt, spacing):
    n = u.shape[0] * u.shape[1]
    uV = u.ravel()
    I = sparse.eye(n)
    A = (theta * dt * 1j) * F + I
    b = (((theta - 1) * dt * 1j) * F + I).dot(uV)

    uN = linalg.spsolve(A, b)
    uN = sp.reshape(uN, u.shape)
    return normalize(uN, spacing)

if __name__ == '__main__':

    # Potential well parameters definition
    v0 = 1000.0
    vM = 200.0
    b = 1.0
    a = 1.0
    R = 0.5
    x0 = -0.5
    y0 = 0.0
    d = 0.25
    x1 = x0 + 2.0*R + d

    # Box definition
    xyMin = -3.0
    xyMax = 3.0
    xyT = 1.0
    dxy = 0.03
    X, Y = sp.mgrid[xyMin:xyMax:dxy, xyMin:xyMax:dxy]

    V = potential_well(X, Y, x0, y0, x1, R, v0) + absorving_borders_box(X, Y, xyT, xyMax, vM)
    pl.figure()
    pl.contourf(X, Y, abs(V), levels = sp.linspace(abs(V).min(), abs(V).max(), 1000))
    pl.colorbar()
    pl.show()
