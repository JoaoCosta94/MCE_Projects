__author__ = 'JoaoCosta'

import scipy as sp
import pyopencl as cl
import platform

# Save files Folder path
if (platform.system() == 'Windows'):
    sPath = 'sources\\'
else:
    sPath = 'sources/'

def getDeciveContext():
    ctx = cl.create_some_context()
    return (ctx, cl.CommandQueue(ctx), cl.mem_flags)

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
    f1 = open(sPath + "precode.cl", "r")
    f2 = open(sPath + "kernel.cl", "r")
    f3 = open(sPath + "source.cl",'w+')
    precode = f1.read()
    kernel = f2.read()
    f3.write(precode + constants + kernel)
    f1.close()
    f2.close()
    f3.close()
    return sPath + "source.cl"

def source_build(ctx, source):
    return cl.Program(ctx, source).build()

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

def reset_device_variables(queue, p_d, A_d, OC_d, k_d, ps_d, pm_d, p_h, A_h, OC_h):
    """
    This function resets the gpu variables to run another simulation
    """
    cl.enqueue_copy(queue, p_d, p_h)
    cl.enqueue_copy(queue, A_d, A_h)
    cl.enqueue_copy(queue, OC_d, OC_h)
    cl.enqueue_copy(queue, k_d, sp.empty_like(p_h))
    cl.enqueue_copy(queue, ps_d, sp.empty_like(p_h))
    cl.enqueue_copy(queue, pm_d, sp.empty_like(p_h))

def bufferCopy(queue, destination, source):
    cl.enqueue_copy(queue, destination, source)