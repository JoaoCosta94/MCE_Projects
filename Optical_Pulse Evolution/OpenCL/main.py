__author__ = 'JoaoCosta'

import scipy as sp
import pylab as pl
import pyopencl as cl
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

def device_code(N, dx, dt, P0, DELTA, GAMA, OC):
    """
    This function generates the source code for the device solver
    """
    # Writing the source code with the constants declared by the user
    constants = ""
    constants = "constant int N=" + str(N) + "; \n"
    constants += "constant float dx=" + str(dx) + "; \n"
    constants += "constant float dt=" + str(dt) + "; \n"
    constants += "constant float P0=" + str(P0) + "; \n"
    constants += "constant float DELTA=" + str(DELTA) + "; \n"
    constants += "constant float GAMA=" + str(GAMA) + "; \n"
    constants += "constant float OC=" + str(OC) + "; \n"
    f1 = open("precode.cl", "r")
    f2 = open("kernel.cl", "r")
    f3 = open("source.cl",'w+')
    precode = f1.read()
    kernel = f2.read()
    f3.write(precode + constants + kernel)
    f1.close()
    f2.close()
    f3.close()

def device_allocate(ctx, MF, x_h, A_h):
    """
    This function allocates memory on the device
    """
    x_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=x_h)
    A_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=A_h)
    k_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=sp.empty_like(x_h))
    xs_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=x_h)
    xm_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=x_h)
    return x_d, A_d, k_d, xs_d, xm_d

if __name__ == "__main__":

    # Definition of problem parameters

    # Grid parameters
    gWidth = 100 # atom grid width
    dx = sp.float32(0.01) # atom grid spacing

    # Time parameters
    tInterval = sp.float32(2.5)
    dt = sp.float32(0.01)

    # State density parameters
    P0 = sp.float32(1.0)
    GAMA = sp.float32(1.0)
    OC = sp.float32(1.0)
    DELTA = sp.float32(1.0)

    # Envelope parameters
    a = sp.float32(1.0)
    b = sp.float32(1.0)

    # Generating grids
    X_h = sp.arange(0.0, gWidth, dx)
    T_h = sp.arange(dt, tInterval, dt)
    N = len(X_h)

    # Generating initial states density
    p11_h, p22_h, p33_h, p21_h, p31_h, p32_h = initial_state(N)
    p_h = sp.array([p11_h, p22_h, p33_h, p21_h, p31_h, p32_h]).T.astype(sp.complex64)

    # Generating initial envelope status
    A_h = 100.0*(sp.random.random_sample(N) + 1j*sp.random.random_sample(N)).astype(sp.complex64)

    # Preparing GPU code
    device_code(N, dx, dt, P0, DELTA, GAMA, OC)

    # Initialization of the device and workspace
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    MF = cl.mem_flags

    # Allocating gpu variables
    p_d, A_d, k_d, ps_d, pm_d= device_allocate(ctx, MF, p_h, A_h)
    W = sp.uint32(6) # The row width to compute the index inside the kernel

    # Loading the source
    f = open("source.cl", "r")
    source = f.read()
    f.close()
    prg = cl.Program(ctx, source).build()

    for t in T_h:
        print t / tInterval * 100
        evolveSate = prg.RK4Step(queue, (N,), None, p_d, A_d, k_d, ps_d, pm_d, W, t)
        evolveSate.wait()
