__author__ = 'JoaoCosta'

import scipy as sp
import pylab as pl
import time

def initial_state(N):
    """
    This function generates the initial state of the N atoms
    """
    p11 = sp.random.random_sample(N) + 1j*sp.random.random_sample(N) #
    p22 = sp.random.random_sample(N) + 1j*sp.random.random_sample(N) #
    p33 = sp.random.random_sample(N) + 1j*sp.random.random_sample(N) # Creation of initial states of the 3 state atoms
    p21 = sp.random.random_sample(N) + 1j*sp.random.random_sample(N) #
    p31 = sp.random.random_sample(N) + 1j*sp.random.random_sample(N) #
    p32 = sp.random.random_sample(N) + 1j*sp.random.random_sample(N) #
    return p11, p22, p33, p21, p31, p32

def evolve_state(A, p):
    """
    This function evolves the states according to the equation system
    """
    # p11 = p[:, 0]
    # p22 = p[:, 1]
    # p33 = p[:, 2]
    # p21 = p[:, 3]
    # p31 = p[:, 4]
    # p32 = p[:, 5]
    p11 = GAMA*(p[:, 1]+sp.conj(p[:, 1]))/2.0 + 1.0j*P0*A*(p[:, 3]+sp.conj(p[:, 3]))
    p22 = -GAMA*(p[:, 1]+sp.conj(p[:, 1])) - 1.0j*(P0*A*(p[:, 3]+sp.conj(p[:, 3])) - OC*(p[:, 5]+sp.conj(p[:, 5])))
    p33 = GAMA*(p[:, 1]+sp.conj(p[:, 1]))/2.0 - 1.0j*OC*(p[:, 5]+sp.conj(p[:, 5]))
    p21 = 1.0j*(P0*A*(p[:, 0]-p[:, 1]) + OC*p[:, 4] - DELTA*p[:, 3]) - GAMA*p[:, 3]
    p31 = 1.0j*(-P0*A*p[:, 5] + OC*p[:, 3] + DELTA*p[:, 4])
    p32 = 1.0j*(-P0*A*p[:, 4] + OC*(p[:, 1]-p[:, 2])) - GAMA*p[:, 5]
    aux = []
    # Rearranging p
    for i in range(len(p11)):
        aux.append(sp.array([p11[i], p22[i], p33[i], p21[i], p31[i], p32[i]]))
    aux = sp.array(aux).astype(complex)
    return aux

def RK4_STEP(dt, A, p):
    """
    This functions evolves the state of the atoms p
    """
    k1 = evolve_state(A, p)
    k2 = evolve_state(A, p + k1*dt/2.0)
    k3 = evolve_state(A, p + k2*dt/2.0)
    k4 = evolve_state(A, p + k3*dt)
    return p + dt/6.0 * (k1 + 2.0*(k2 + k3) + k4)

def evolve_envelope(A, p21, X, t):
    """
    This function evolves the envelope
    """
    aux = eps*(sp.roll(A, 1) + sp.roll(A, -1)) + gama*(p21*sp.exp(1j*(Kp*X - Wp*t)) + CC)
    return A + 1j*aux

if __name__ == '__main__':

    # Problem parameters
    global GAMA
    global P0
    global OC
    global DELTA
    global eps
    global gama
    global Kp
    global Wp
    global CC
    GAMA = 1.0
    P0 = 1.0
    OC = 1.0
    DELTA = 1.0

    # Envelope parameters
    a = 1.0
    b = 1.0
    CC = 0.0 # no direct current
    Kp = 1.0
    Wp = 1.0

    # Grid parameters and definition
    width = 100 # atom chain width
    dx = 0.01 # atom chain spacing
    X = sp.arange(0.0, width, dx) # atom grid

    # Time parameters and definition
    interval = 10.0 # simulation duration
    dt = 0.01 # time step
    T = sp.arange(dt, interval, dt) # time array

    # Auxiliary variables
    eps = a*dt/dx**2
    gama = b*dt*P0

    # Generation of initial distributions
    N = len(X) # number of atoms according to grid definition
    p11, p22, p33, p21, p31, p32 = initial_state(N) # Creation of initial state for each atom
    p = []
    for i in range (N):
        p.append(sp.array([p11[i], p22[i], p33[i], p21[i], p31[i], p32[i]]))
    p = sp.array(p).astype(complex)

    # TODO: Create a real envelope later like a cosine
    A = 100.0*(sp.random.random_sample(N) + 1j*sp.random.random_sample(N)) # Initial envelope generated as random dist

    print 'All calculations will be performed using only python'

    p21_evolution = []
    A_evolution = []
    start = time.time()
    for t in T:
        # Evolve envelope
        A = evolve_envelope(A, p[:, 3], X, t)
        # For each time instant evolve the state with the new values of A
        p = RK4_STEP(dt, A, p)
        # Append to mesh
        A_evolution.append(A)
        p21_evolution.append(abs(p[3]))

    tCalc = time.time() - start
    print 'Calculations took ' + str(tCalc) + ' seconds'

    # Plotting preparation and presentation
    X_MESH, T_MESH = sp.meshgrid(X, T)
    p21_evolution = sp.array(p21_evolution)
    A_evolution = sp.array(A_evolution)

    # p21 evolution
    pl.figure()
    pl.xlabel('X')
    pl.ylabel('t')
    pl.contourf(X_MESH, T_MESH, p21_evolution, levels = sp.linspace(p21_evolution.min(), p21_evolution.max(), 100))
    pl.colorbar()

    # Envelope evolution
    pl.figure()
    pl.xlabel('X')
    pl.ylabel('t')
    pl.contourf(X_MESH, T_MESH, abs(A_evolution), levels = sp.linspace(A_evolution.min(), A_evolution.max(), 100))
    pl.colorbar()

    pl.show()