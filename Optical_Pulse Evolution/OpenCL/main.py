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

def device_allocate(ctx, MF, p_shape, A_shape):
    """
    This function allocates memory on the device
    """
    p_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=sp.zeros(p_shape))
    A_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=sp.zeros(A_shape).astype((sp.complex64)))
    OC_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=sp.zeros(A_shape))
    k_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=sp.zeros(p_shape))
    ps_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=sp.zeros(p_shape))
    pm_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=sp.zeros(p_shape))
    return p_d, A_d, OC_d, k_d, ps_d, pm_d

def plotting_pulse(iPath, P0, OC, X, T, evolution):
    """
    This function plots the amplitude of the pulse's envelope over time on all the grid positions
    """
    xGrid, tGrid = sp.meshgrid(X, T)
    pl.figure("Pulse_P0=" + str(P0) + "_OC=" + str(OC))
    pl.title("Pulse propagation W/ " + "P0=" + str(P0) + ", OC=" + str(OC))
    pl.xlabel("x")
    pl.ylabel("t")
    pl.contourf(xGrid, tGrid, evolution, levels=sp.linspace(0, 1.5 * A0, 30))
    # pl.contourf(xGrid, tGrid, evolution)
    pl.colorbar()
    pl.savefig(iPath+"Pulse_P0_"+str(P0)+"_OC_"+str(OC)+".png")

def plotting_state(iPath, P0, OC, X, T, state11, state22, state33):
    """
    This function plots the state densities over time on all the grid positions
    """
    xGrid, tGrid = sp.meshgrid(X, T)
    pl.figure("p11_P0=" + str(P0) + "_OC=" + str(OC))
    pl.title("State 11 evolution W/ " + "P0=" + str(P0) + ", OC=" + str(OC))
    pl.xlabel("x")
    pl.ylabel("t")
    pl.contourf(xGrid, tGrid, state11, sp.linspace(0.0, state11.max(), 30))
    pl.colorbar()
    pl.savefig(iPath+"p11_P0_"+str(P0)+"_OC_"+str(OC)+".png")

    pl.figure("p22_P0=" + str(P0) + "_OC=" + str(OC))
    pl.title("State 22 evolution W/ " + "P0=" + str(P0) + ", OC=" + str(OC))
    pl.xlabel("x")
    pl.ylabel("t")
    pl.contourf(xGrid, tGrid, state22, sp.linspace(0.0, state22.max(), 30))
    pl.colorbar()
    pl.savefig(iPath+"p22_P0_"+str(P0)+"_OC_"+str(OC)+".png")

    pl.figure("p33_P0=" + str(P0) + "_OC=" + str(OC))
    pl.title("State 33 evolution W/ " + "P0=" + str(P0) + ", OC=" + str(OC))
    pl.xlabel("x")
    pl.ylabel("t")
    pl.contourf(xGrid, tGrid, state33, sp.linspace(0.0, state33.max(), 30))
    pl.colorbar()
    pl.savefig(iPath+"p33_P0_"+str(P0)+"_OC_"+str(OC)+".png")

def reset_device_variables(p_d, A_d, OC_d, k_d, ps_d, pm_d, p_h, A_h, OC_h):
    """
    This function resets the gpu variables to run another simulation
    """
    cl.enqueue_copy(queue, p_d, p_h)
    cl.enqueue_copy(queue, A_d, A_h)
    cl.enqueue_copy(queue, OC_d, OC_h)
    cl.enqueue_copy(queue, k_d, sp.empty_like(p_h))
    cl.enqueue_copy(queue, ps_d, sp.empty_like(p_h))
    cl.enqueue_copy(queue, pm_d, sp.empty_like(p_h))

def simulation_run(P0, OC, N, W, fPath, iPath, X_h, T_h, prg, p_d, A_d, OC_d, k_d, ps_d, pm_d):
    """
    This function runs the simulation on the gpu
    saves the results and
    plots the graph
    with a different set of parameters
    """
    # Save values interval & variables
    snapMultiple = 100
    pulseEvolution = [abs(A_h)**2]
    p11Evolution = [abs(p_h[:, 0])]
    p22Evolution = [abs(p_h[:, 1])]
    p33Evolution = [abs(p_h[:, 2])]
    tInstants = [0.0]

    pulsePath = fPath+'Pulse_Evol_'+str(N)+'_oc_'+str(OC)+'_p0_'+str(P0)+'.npy'
    p11Path = fPath+'P11_Evol_'+str(N)+'_oc_'+str(OC)+'_p0_'+str(P0)+'.npy'
    p22Path = fPath+'P22_Evol_'+str(N)+'_oc_'+str(OC)+'_p0_'+str(P0)+'.npy'
    p33Path = fPath+'P33_Evol_'+str(N)+'_oc_'+str(OC)+'_p0_'+str(P0)+'.npy'
    tPath = fPath+'T_'+str(N)+'_oc_'+str(OC)+'_p0_'+str(P0)+'.npy'
    xPath = fPath+'X_'+str(N)+'_oc_'+str(OC)+'_p0_'+str(P0)+'.npy'

    for i in range(len(T_h)):
        # Evolve State
        evolveSate = prg.RK4Step(queue, (N,), None, p_d, A_d, OC_d, k_d, ps_d, pm_d, W, T_h[i])
        evolveSate.wait()
        # Evolve Pulse
        evolvePulse = prg.PulseEvolution(queue, (N,), None, p_d, A_d, T_h[i], W)
        evolvePulse.wait()
        if (i % snapMultiple == 0):
            # Grabbing snapshot instant
            tInstants.append(T_h[i])
            # Copying state to RAM
            cl.enqueue_copy(queue, p_h, p_d)
            p11Evolution.append(abs(p_h[:, 0]))
            p22Evolution.append(abs(p_h[:, 1]))
            p33Evolution.append(abs(p_h[:, 2]))
            # Copying pulse to RAM
            cl.enqueue_copy(queue, A_h, A_d)
            pulseEvolution.append(abs(A_h)**2)
            print 'Snapshot Saved'
        print "{:.3f}".format(T_h[i] / tInterval * 100) + '%'

    # Converting to arrays
    pulseEvolution = sp.array(pulseEvolution)
    p11Evolution = sp.array(p11Evolution)
    p22Evolution = sp.array(p22Evolution)
    p33Evolution = sp.array(p33Evolution)
    tInstants = sp.array(tInstants)

    # Saving data to files
    sp.save(pulsePath, pulseEvolution)
    sp.save(p11Path, p11Evolution)
    sp.save(p22Path, p22Evolution)
    sp.save(p33Path, p33Evolution)
    sp.save(tPath, tInstants)
    sp.save(xPath, X_h)

    plotting_pulse(iPath, P0, OC, X_h, tInstants, pulseEvolution)
    plotting_state(iPath, P0, OC, X_h, tInstants, p11Evolution, p22Evolution, p33Evolution)


if __name__ == "__main__":

    global A0

    # Definition of problem parameters
    a0 = 1.0
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
    P0_List = sp.array([1.0]).astype(sp.float32)
    DELTA = sp.float32(1.0)
    GAMA = (1.0*DELTA).astype(sp.float32)
    OC_List = sp.array([0.5]).astype(sp.float32)

    # Envelope parameters
    A0 = 1.0
    a = sp.float32(1.0)
    b = sp.float32(1.0)
    disp = gWidth/4.0
    iWidth = 10.0
    k = 0.5*sp.float32(len(T_h)/len(X_h))

    # Polarization parameters
    Kp = sp.float32(1.0)
    Wp = sp.float32(1.0)


########################################################################################################################

    # Initialization of the device and workspace
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    MF = cl.mem_flags

    # Allocating gpu variables
    W = sp.uint32(6) # The row width to compute the index inside the kernel
    p_shape = (N, W)
    A_shape = X_h.shape
    p_d, A_d, OC_d, k_d, ps_d, pm_d= device_allocate(ctx, MF, p_shape, A_shape)

########################################################################################################################

    # Save files Folder path
    if (platform.system() == 'Windows'):
        fPath = 'Data_Files\\'
        iPath = 'Images\\'
    else:
        fPath = 'Data_Files/'
        iPath = 'Images/'

########################################################################################################################

    print 'All calculations will be performed using OpenCL sweet sweet magic'

    start = time.time()
    for P0 in P0_List:
        for OC in OC_List:

            # System Constants
            EPS = sp.float32(a * dt/dx**2)
            G = sp.float32(b*dt*P0)
            CC = sp.float32(0.0)

            # Generating initial states density
            p11_h, p22_h, p33_h, p21_h, p31_h, p32_h = initial_state(N)
            p_h = []
            for i in range (N):
                p_h.append(sp.array([p11_h[i], p22_h[i], p33_h[i], p21_h[i], p31_h[i], p32_h[i]]))
            p_h = sp.array(p_h).astype(sp.complex64)

            # Generating initial envelope status
            A_h = (A0 * sp.exp(-((X_h-disp)/iWidth)**2)*sp.exp(-1j*k*X_h)).astype(sp.complex64)
            OC_h = OC*sp.ones(X_h.shape).astype(sp.float32)

            # Reset device variables
            reset_device_variables(p_d, A_d, OC_d, k_d, ps_d, pm_d, p_h, A_h, OC_h)

            # Preparing GPU code
            device_code(N, dx, dt, P0, DELTA, GAMA, EPS, G, Kp, Wp, CC)

            # Loading the source
            f = open("source.cl", "r")
            source = f.read()
            f.close()
            prg = cl.Program(ctx, source).build()

            simulation_run(P0, OC, N, W, fPath, iPath, X_h, T_h, prg, p_d, A_d, OC_d, k_d, ps_d, pm_d)

    tCalc = time.time() - start
    print 'Calculations took ' + str(tCalc) + ' seconds'