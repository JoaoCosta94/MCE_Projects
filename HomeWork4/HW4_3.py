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
    border[idx] -= vM * ((abs(X[idx]) - xyMax + xyT)**2 * 1j + (abs(X[idx]) - xyMax + xyT)**2)
    border[idy] -= vM * ((abs(Y[idy]) - xyMax + xyT)**2 * 1j + (abs(Y[idy]) - xyMax + xyT)**2)
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

def hamiltonian_operator(X, Y, spacing, xyT, xyMax, x0, y0, x1, R, v0, vM):
    """
    This function generates the Hamiltonian Operator matrix with given potential and absorbing borders box
    """
    L = lap(X.shape, spacing)
    V = potential_well(X, Y, x0, y0, x1, R, v0) + absorving_borders_box(X, Y, xyT, xyMax, vM)
    return -L + sparse.diags(V.ravel(),0, format = 'dia')

def theta_family_step(F, u, theta, dt, spacing):
    """
    This function evolves the state by a time step using the theta family method
    Crank-Nicolson is being used
    """
    n = u.shape[0] * u.shape[1]
    uV = u.ravel()
    I = sparse.eye(n)
    A = (theta * dt * 1j) * F + I
    b = (((theta - 1) * dt * 1j) * F + I).dot(uV)

    uN = linalg.spsolve(A, b)
    uN = sp.reshape(uN, u.shape)
    return normalize(uN, spacing)

def well_points(X, Y, x0, y0, R):
    """
    This function determines which grid points are inside the well
    """
    return sp.where((X-x0)**2 + (Y-y0)**2 < R**2)

def prob_ratio(prob, id):
    """
    This function calculates the ratio of probability inside the well
    """
    return sum(prob[id])

def simulate_ssfm(X, Y, psi, V, Wx, Wy, time, dt, id):
    """
    This function performs the simulation using split step Fourier  method
    """
    # pl.ion()
    # pl.show(block = False)
    tunnel = []
    for t in time:
        # Probability density
        prob = psi.real**2 + psi.imag**2
        tunnel.append(prob_ratio(prob, id))

    #     pl.figure()
    #     pl.title('t = '+str(t))
    #     pl.contourf(X, Y, prob, levels = sp.linspace(0.0, prob.max(), 100))
    #     pl.colorbar()
    #     pl.contour(X, Y, V.real)
    #     # pl.draw()
    #     # pl.clf()

        psi = split_step_fourier(psi, V, Wx, Wy, dt)

    # pl.show()

    return sp.array(tunnel)

def simulate_cn(X, Y, psi, V, H, time, dt, id):
    """
    This function performs the simulation using Crank-Nicolson method
    """
    # pl.ion()
    # pl.show(block = False)
    tunnel = []
    for t in time:
        # Probability density
        prob = psi.real**2 + psi.imag**2
        tunnel.append(prob_ratio(prob, id))

    #     pl.figure()
    #     pl.title('t = '+str(t))
    #     pl.contourf(X, Y, prob, levels = sp.linspace(0.0, prob.max(), 100))
    #     pl.colorbar()
    #     pl.contour(X, Y, V.real)
    #     # pl.draw()
    #     # pl.clf()

        psi = theta_family_step(H, psi, 0.5, dt, dxy)

    # pl.show()

    return sp.array(tunnel)

def simulation(v0, x0, y0, d, R, xyMin, xyMax, dxy, xyT, vM, method = 'SSFM'):

    x1 = x0 + 2.0*R + d

    # Grid definition
    X, Y = sp.mgrid[xyMin:xyMax:dxy, xyMin:xyMax:dxy]

    # Potential definition
    V = potential_well(X, Y, x0, y0, x1, R, v0) + absorving_borders_box(X, Y, xyT, xyMax, vM)
    id = well_points(X, Y, x1, y0, R)

    # Initial state definition
    psi = initial_state(x0, y0, X, Y)
    psi = normalize(psi, dxy)

    # Time parameters definition
    tMax = 0.1
    dt = .001
    time = sp.arange(0.0, tMax+dt, dt)

    if method == 'SSFM':
        print 'Simulating with split step Fourier method'
        # Simulation ran using split step Fourier method
        # Definition of Fourier space (FFT space) frequencies
        Wx, Wy = w_frequencies(psi, dxy)
        return time, simulate_ssfm(X, Y, psi, V, Wx, Wy, time, dt, id)
    else:
        print 'Simulating with Crank-Nicolson method'
        # Simulation ran using Crank-Nicolson method
        # Definition of the Hamiltonian operator matrix
        H = hamiltonian_operator(X, Y, dxy, xyT, xyMax, x0, y0, x1, R, v0, vM)

        # Simulation
        return time, simulate_cn(X, Y, psi, V, H, time, dt, id)

if __name__ == '__main__':

    # Potential well parameters definition
    v0 = 100.0
    vM = 200.0
    R = 0.5
    x0 = -0.5
    y0 = 0.0
    d_array = sp.linspace(0.0001, 0.1, 20)

    # Box definition
    xyMin = -2.0
    xyMax = 2.0
    xyT = 2.0*xyMax/3.0
    dxy = 0.01

    time_mesh = []
    tunnel_mesh = []
    for d in d_array:
        time, tunnel = simulation(v0, x0, y0, d, R, xyMin, xyMax, dxy, xyT, vM)
        pl.figure()
        pl.title('Tunnel effect for d = '+str(d))
        pl.xlabel('Time')
        pl.ylabel('Probability inside the second potential well')
        pl.scatter(time, tunnel)

    pl.show()

