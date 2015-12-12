__author__ = 'JoaoCosta'

import scipy as sp
from scipy.sparse import linalg
from scipy import sparse
from scipy import sum
import pylab as pl
from time import time

def potential_well(X, Y, x0, y0, R, v0):
    """
    This function generates the potential well
    """
    V = v0 * ((X-x0)**2 + (Y-y0)**2 > R**2)
    return V

def absorving_borders_box(X, Y, xyL, xyMax, vM):
    """
    This function generates the absorbing potential on the borders
    """
    x = abs(X)
    y = abs(Y)
    B = sp.zeros(X.shape)
    id = sp.where(abs(X) - xyL < xyMax)
    B[id] += vM
    id = sp.where(abs(Y) -xyL < xyMax)
    B[id] += vM
    return B

def lap(shape, spacing):
    """
    This function generates the laplacian operator
    """
    n = shape[0]*shape[1]
    return ( -4.*sparse.eye(n, n, 0) + sparse.eye(n, n, 1) + sparse.eye(n, n, -1) + sparse.eye(n, n, shape[1]) + sparse.eye(n, n, -shape[1]) )/spacing**2

def initial_state(k, theta, x0, X, Y):
    """
    This function generates the initial state with given parameters
    """
    kx = k*sp.cos(theta)
    ky = k*sp.sin(theta)
    xi = 0.0 #(R + x0) / 2.0
    delta = 0.1 #(R-x0) / 20.0
    psi = sp.exp(1j*(kx*X + ky*Y))*sp.exp(-((X-xi)**2 + (Y-y0)**2) / delta**2)
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


if __name__ == '__main__':

    # Potential well parameters definition
    v0 = 1000.0
    b = 1.0
    a = 1.0
    R = 1.0
    x0 = 0.0 #- R / 2.0
    y0 = 0.0

    # Box definition
    xyMin = -3.0
    xyMax = 3.0
    xyL = 1.0
    dxy = 0.03
    X, Y = sp.mgrid[xyMin:xyMax:dxy, xyMin:xyMax:dxy]

    # Gaussian state definition
    k = 1000.0
    theta = 0.0
    psi = initial_state(k, theta, x0, X, Y)

    # Normalization
    psi = normalize(psi, dxy)

    # Time parameters definition
    tMax = 10.0
    dt = .001
    time = sp.arange(dt, tMax+dt, dt)

    # Potential
    V = potential_well(X, Y, x0, y0, R, v0) # + absorving_borders_box(X, Y, xyL, 1000)
    # V = absorving_borders_box(X, Y, xyL, xyMax, 1000)

    pl.figure()
    pl.contour(X, Y, V.real)
    pl.show()

    prob = psi.real**2 + psi.imag**2

    pl.ion()
    pl.contourf(X, Y, prob, levels = sp.linspace(0.0, prob.max(), 100))
    pl.colorbar()
    pl.contour(X, Y, V.real)
    pl.draw()

    for t in time:

        psi = split_step_fourier(psi, V, dxy, dt)
        prob = psi.real**2 + psi.imag**2

        pl.clf()
        # pl.figure('t = ' + str(t))
        pl.contourf(X, Y, prob, levels = sp.linspace(0.0, prob.max(), 100))
        pl.colorbar()
        pl.contour(X, Y, V.real)
        pl.draw()