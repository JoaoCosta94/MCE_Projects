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

def absorving_borders_box(X, Y, xyT, xyMax, vM):
    """
    This function generates the absorbing potential on the borders
    """
    border = sp.zeros(X.shape, dtype = complex)
    idx = sp.where(abs(X) > (xyMax - xyT))
    idy = sp.where(abs(Y) > (xyMax - xyT))
    border[idx] -= vM * ((abs(X[idx]) - xyMax + xyT)**2 * 1j + (abs(X[idx]) - xyMax + xyT)**2)
    border[idy] -= vM * ((abs(Y[idy]) - xyMax + xyT)**2 * 1j + (abs(Y[idy]) - xyMax + xyT)**2)
    return border

def initial_state(k, theta, xi, yi, X, Y):
    """
    This function generates the initial state with given parameters
    """
    kx = k*sp.cos(theta)
    ky = k*sp.sin(theta)
    delta = 0.1 #(R-x0) / 20.0
    psi = sp.exp(1j*(kx*X + ky*Y))*sp.exp(-((X-xi)**2 + (Y-yi)**2) / delta**2)
    return psi

def normalize(state, spacing):
    """
    This function normalizes a given state
    """
    N = sp.sqrt(sum(abs(state)**2))
    return state / N

def w_frequencies(state, spacing):
    """
    This function calculates the FFT frequencies
    """
    nX = state.shape[0]
    nY = state.shape[1]
    return sp.meshgrid(2.0 * sp.pi * pl.fftfreq(nX, spacing), 2.0 * sp.pi * pl.fftfreq(nY, spacing))

def split_step_fourier(state, V, Wx, Wy, dt):
    """
    This function evolves the state by a time step using the split step Fourier method
    """
    stateNew = sp.exp(-1j * dt * V) * state
    stateNew = pl.fft2(stateNew)
    stateNew = pl.exp(-1j * dt * (Wx**2 + Wy**2)) * stateNew

    return pl.ifft2(stateNew)

def simulate_ssfm(X, Y, psi, V, Wx, Wy, time, dt, id):
    """
    This function performs the simulation using split step Fourier  method
    """
    pl.ion()
    pl.show(block = False)
    for t in time:
        # Probability density
        prob = psi.real**2 + psi.imag**2

    #     pl.figure()
    #     pl.title('t = '+str(t))
        pl.contourf(X, Y, prob, levels = sp.linspace(0.0, prob.max(), 100))
        pl.colorbar()
        pl.contour(X, Y, V.real)
        pl.draw()
        pl.clf()

        psi = split_step_fourier(psi, V, Wx, Wy, dt)

    # pl.show()

    return 0

def simulation(v0, x0, y0, R, xi, yi, xyMin, xyMax, dxy, xyT, vM, k, theta, method = 'SSFM'):

    # Grid definition
    X, Y = sp.mgrid[xyMin:xyMax:dxy, xyMin:xyMax:dxy]

    # Potential definition
    V = potential_well(X, Y, x0, y0, R, v0) + absorving_borders_box(X, Y, xyT, xyMax, vM)

    # Initial state definition
    psi = initial_state(k, theta, xi, yi, X, Y)
    psi = normalize(psi, dxy)
    # prob = psi.real**2 + psi.imag**2

    # pl.figure()
    # pl.contourf(X, Y, prob, levels = sp.linspace(0.0, prob.max(), 100))
    # pl.colorbar()
    # pl.contour(X, Y, V.real)
    # pl.show()

    #Time parameters definition
    tMax = 0.1
    dt = .001
    time = sp.arange(0.0, tMax+dt, dt)

    print 'Simulating with split step Fourier method'
    # Simulation ran using split step Fourier method
    # Definition of Fourier space (FFT space) frequencies
    Wx, Wy = w_frequencies(psi, dxy)
    return time, simulate_ssfm(X, Y, psi, V, Wx, Wy, time, dt, id)

if __name__ == '__main__':

    # Potential well parameters definition
    v0 = 500.0
    vM = 1000.0
    R = 0.5
    x0 = 0.0
    y0 = 0.0

    # Box definition
    xyMin = -3.0
    xyMax = 3.0
    xyT = 1.0
    dxy = 0.01

    # Gaussian state definition
    k = 50.0
    theta = 2.4 * sp.pi / 4.0
    xi = 1.0
    yi = -1.0

    simulation(v0, x0, y0, R, xi, yi, xyMin, xyMax, dxy, xyT, vM, k, theta)