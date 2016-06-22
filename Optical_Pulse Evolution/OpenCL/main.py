__author__ = 'JoaoCosta'

import scipy as sp
import time
import device
import simulation

if __name__ == "__main__":

    # Grid parameters
    gWidth = 100 # atom grid width
    dx = sp.float32(0.1) # atom grid spacing

    # Time parameters
    tInterval = sp.float32(10.0)
    dt = sp.float32(0.0001)

    # Generating grids
    X_h = sp.arange(0.0, gWidth+dx, dx).astype(sp.float32)
    T_h = sp.arange(dt, tInterval+dt, dt).astype(sp.float32)
    N = len(X_h)

    # State density parameters
    P0_List = sp.array([0.5]).astype(sp.float32)
    DELTA = sp.float32(1.0)
    GAMA = (1.0*DELTA).astype(sp.float32)
    OC_List = sp.array([1.0]).astype(sp.float32)

    # Envelope parameters
    A0 = 1.0
    a = sp.float32(1.0)
    b = sp.float32(1.0)
    disp = gWidth/4.0
    iWidth = 10.0
    # k = 0.5*sp.float32(len(T_h)/len(X_h))
    k = 1000.0

    # Polarization parameters
    Kp = sp.float32(1.0)
    Wp = sp.float32(1.0)

########################################################################################################################

    # Initialization of the device and workspace
    ctx, queue, MF = device.getDeciveContext()

    # Allocating gpu variables
    W = sp.uint32(6) # The row width to compute the index inside the kernel
    p_shape = (N, W)
    A_shape = X_h.shape
    p_d, A_d, OC_d, k_d, ps_d, pm_d= device.device_allocate(ctx, MF, p_shape, A_shape)

########################################################################################################################

    print 'All calculations will be performed using OpenCL sweet sweet magic'

    start = time.time()
    for P0 in P0_List:
        for OC in OC_List:

            print "Simulating " + " P0=" + str(P0) + ", OC=" + str(OC)

            # System Constants
            EPS = sp.float32(a * dt/dx**2)
            G = sp.float32(b*dt*P0)
            CC = sp.float32(0.0)

            # Generating initial states density
            p11_h, p22_h, p33_h, p21_h, p31_h, p32_h = simulation.initial_state(N)
            p_h = []
            for i in range (N):
                p_h.append(sp.array([p11_h[i], p22_h[i], p33_h[i], p21_h[i], p31_h[i], p32_h[i]]))
            p_h = sp.array(p_h).astype(sp.complex64)

            # Generating initial envelope status
            A_h = (A0 * sp.exp(-((X_h-disp)/iWidth)**2)*sp.exp(-1j*k*X_h)).astype(sp.complex64)
            OC_h = OC*sp.ones(X_h.shape).astype(sp.float32)

            # Reset device variables
            device.reset_device_variables(queue, p_d, A_d, OC_d, k_d, ps_d, pm_d, p_h, A_h, OC_h)

            # Preparing GPU code
            sPath = device.device_code(N, dx, dt, P0, DELTA, GAMA, EPS, G, Kp, Wp, CC)

            # Loading the source
            f = open(sPath, "r")
            source = f.read()
            f.close()
            prg = device.source_build(ctx, source)

            simulation.simulation_run(queue, A0, P0, OC, N, W, X_h, T_h, tInterval,
                                      prg, A_h, p_h, p_d, A_d, OC_d, k_d, ps_d, pm_d)

    tCalc = time.time() - start
    print 'Calculations took ' + str(tCalc) + ' seconds'