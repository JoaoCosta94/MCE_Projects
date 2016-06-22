__author__ = 'JoaoCosta'

import scipy as sp
import pyopencl as cl
import platform
import device
import plotting

# Save files Folder path
if (platform.system() == 'Windows'):
    fPath = 'Data_Files\\'
    iPath = 'Images\\'
else:
    fPath = 'Data_Files/'
    iPath = 'Images/'

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

def simulation_run(queue, A0, P0, OC, N, W, X_h, T_h, tInterval, prg, A_h, p_h, p_d, A_d, OC_d, k_d, ps_d, pm_d):
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
            device.bufferCopy(queue, p_h, p_d)
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

    # Generate and save respective graphs
    pulseScale = sp.linspace(0.0, 1.5 * A0, 100)
    plotting.pulse_graph(P0, OC, X_h, T_h, pulseEvolution, pulseScale)
    plotting.state_graph(P0, OC, X_h, T_h, p11Evolution, p22Evolution, p33Evolution)