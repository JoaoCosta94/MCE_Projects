__author__ = 'JoaoCosta'

import scipy as sp
import pylab as pl
import pyopencl as cl
import time
import platform

def initial_state(N):
    """
    This function generates the initial state of the N atoms
    """
    # TODO: Change pxy to values that make sense (p11+p22+p33=1)
    p11 = sp.ones(N)  #
    p22 = sp.zeros(N) #
    p33 = sp.zeros(N) # Creation of initial states of the 3 state atoms
    p21 = sp.zeros(N) #
    p31 = sp.zeros(N) #
    p32 = sp.zeros(N) #
    return p11, p22, p33, p21, p31, p32

def device_code(N, dx, dt, P0, DELTA, GAMA, EPS, G, Kp, Wp, CC):
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
    constants += "constant float EPS=" + str(EPS) + "; \n"
    constants += "constant float G=" + str(G) + "; \n"
    constants += "constant float Kp=" + str(Kp) + "; \n"
    constants += "constant float Wp=" + str(Wp) + "; \n"
    constants += "constant float CC=" + str(CC) + "; \n"
    f1 = open("precode.cl", "r")
    f2 = open("kernel.cl", "r")
    f3 = open("source.cl",'w+')
    precode = f1.read()
    kernel = f2.read()
    f3.write(precode + constants + kernel)
    f1.close()
    f2.close()
    f3.close()

def device_allocate(ctx, MF, X_h, p_h, A_h, OC_h):
    """
    This function allocates memory on the device
    """
    X_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=X_h)
    p_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=p_h)
    A_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=A_h)
    OC_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=OC_h)
    k_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=sp.empty_like(p_h))
    ps_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=sp.empty_like(p_h))
    pm_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=sp.empty_like(p_h))
    return X_d, p_d, A_d, OC_d, k_d, ps_d, pm_d

if __name__ == "__main__":

    # Definition of problem parameters

    # Grid parameters
    gWidth = 100 # atom grid width
    dx = sp.float32(0.01) # atom grid spacing

    # Time parameters
    tInterval = sp.float32(10.0)
    dt = sp.float32(0.01)

    # Generating grids
    X_h = sp.arange(0.0, gWidth+dx, dx).astype(sp.float32)
    T_h = sp.arange(dt, tInterval+dt, dt).astype(sp.float32)
    N = len(X_h)

    # State density parameters
    # TODO: Change OC to values that make sense
    P0 = sp.float32(1.0)
    GAMA = sp.float32(1.0)
    DELTA = sp.float32(1.0)
    OC_h = sp.ones(X_h.shape).astype(sp.float32)

    # Envelope parameters
    # TODO: Change to values that make sense
    a = sp.float32(1.0)
    b = sp.float32(1.0)
    Kp = sp.float32(1.0)
    Wp = sp.float32(1.0)
    EPS = a * dt/dx**2
    G = b*dt*P0
    CC = sp.float32(0.0)

    # Generating initial states density
    p11_h, p22_h, p33_h, p21_h, p31_h, p32_h = initial_state(N)
    p_h = []
    for i in range (N):
        p_h.append(sp.array([p11_h[i], p22_h[i], p33_h[i], p21_h[i], p31_h[i], p32_h[i]]))
    p_h = sp.array(p_h).astype(complex)

    # Generating initial envelope status
    # TODO: Change A to values that make sense
    # A_h = 100.0*(sp.random.random_sample(N) + 1j*sp.random.random_sample(N)).astype(complex)
    A_h = (sp.arange(N) + 1j*sp.zeros(N)).astype(sp.complex64)
    A_h = (sp.exp(-((A_h-N/2.0)/(0.05 * N))**2)*sp.exp(1j * 200.0 * A_h /N)).astype(complex)

    # Preparing GPU code
    device_code(N, dx, dt, P0, DELTA, GAMA, EPS, G, Kp, Wp, CC)

    # Initialization of the device and workspace
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    MF = cl.mem_flags

    # Allocating gpu variables
    X_d, p_d, A_d, OC_d, k_d, ps_d, pm_d= device_allocate(ctx, MF, X_h, p_h, A_h, OC_h)
    W = sp.uint32(6) # The row width to compute the index inside the kernel

    # Loading the source
    f = open("source.cl", "r")
    source = f.read()
    f.close()
    prg = cl.Program(ctx, source).build()

    # Save values interval & variables
    snapMultiple = 10
    pulseEvolution = [A_h]
    p21Evolution = [p_h[:, 3]]
    tInstants = [0.0]

    # Save files path
    if (platform.system() == 'Windows'):
        fPath = 'Data_Files\\'
    else:
        fPath = 'Data_Files/'
    pulsePath = fPath+'Pulse_Evol_'+str(N)+'_dx_'+str(dx)+'_dt_'+str(dt)+'.npy'
    p21Path = fPath+'P21_Evol_'+str(N)+'_dx_'+str(dx)+'_dt_'+str(dt)+'.npy'
    tPath = fPath+'T_'+str(N)+'_dx_'+str(dx)+'_dt_'+str(dt)+'.npy'
    xPath = fPath+'X_'+str(N)+'_dx_'+str(dx)+'_dt_'+str(dt)+'.npy'

    print 'All calculations will be performed using OpenCL sweet sweet magic'

    start = time.time()
    for i in range(len(T_h)):
        # Evolve State
        evolveSate = prg.RK4Step(queue, (N,), None, p_d, A_d, OC_d, k_d, ps_d, pm_d, W, T_h[i])
        evolveSate.wait()
        # Evolve Pulse
        evolvePulse = prg.PulseEvolution(queue, (N,), None, p_d, A_d, dx, T_h[i], W)
        evolvePulse.wait()
        if (i % snapMultiple == 0):
            # Grabbing snapshot instant
            tInstants.append(T_h[i])
            # Copying state to RAM
            cl.enqueue_copy(queue, p_h, p_d)
            p21Evolution.append(p_h[:, 3])
            # Copying pulse to RAM
            cl.enqueue_copy(queue, A_h, A_d)
            pulseEvolution.append(A_h)
            print 'Snapshot Saved'
        print "{:.2f}".format(T_h[i] / tInterval * 100) + '%'

    # Converting to arrays
    pulseEvolution = sp.array(pulseEvolution)
    p21Evolution = sp.array(p21Evolution)
    tInstants = sp.array(tInstants)

    # Saving data to files
    sp.save(pulsePath, pulseEvolution)
    sp.save(p21Path, p21Evolution)
    sp.save(tPath, tInstants)
    sp.save(xPath, X_h)

    tCalc = time.time() - start
    print 'Calculations & saving to files took ' + str(tCalc) + ' seconds'
