__author__ = 'Joao Costa'

import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import time

"""
Solve the problem
Xi' = M1*Xi + M2*Xi~
M1 and M2 are 6*6 Matrixes which elements are complex numbers
Xi in the form [P11i, P22i, P33i, P21i, P31i, P32i, Ai] where Pxyi is a complex number
""" 

########################################################
#                                                      #
#'_h' buffers are host buffers. '_d' are device buffers#
#                                                      #
########################################################

#Initialization of the device and workspace
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
MF = cl.mem_flags

# Constants 
M = 2000 # Number of atoms
L = np.float32(0.000000001) # Atom Spacing
N = 1000 # Number of time intervals
dt = np.float32(0.1) # Time interval
Timeline = np.arange(0.0, N, dt) .astype(np.float32)
p0 = np.float32(1.0) # constant P0 [OMP = P0*Ai]
delta = np.float32(1.0) # constant DELTA
gama = np.float32(1.0) # constant GAMA
omc = np.float32(1.0) # constant OMC

# Writing the source code with the constants declared by the user
text = ""
##text = "constant int M=" + str(M) + "; \n"
##text += "constant float L=" + str(L) + "; \n"
##text += "constant float dt=" + str(dt) + "; \n"
##text += "constant float p0=" + str(p0) + "; \n"
##text += "constant float delta=" + str(delta) + "; \n"
##text += "constant float gama=" + str(gama) + "; \n"
##text += "constant float omc=" + str(omc) + "; \n"
f1 = open("precode.cl", "r")
f2 = open("kernel_2.cl", "r")
f3 = open("source.cl",'w+')
precode = f1.read()
kernel = f2.read()
f3.write(precode + text + kernel)
f1.close()
f2.close()
f3.close()

#Initial Conditions 
A_h = (np.arange(M) + 1j*np.zeros(M)).astype(np.complex64)
A_h = np.exp(-((A_h-M/2.0)/(0.05 * M))**2)*np.exp(1j * 200.0 * A_h /M)

P11_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P22_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P33_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P21_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P31_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P32_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)

W = np.uint32(7) # The row width to compute the index inside the kernel
X_h = []
for i in range(M):
    X_h.append( np.array([P11_h[i], P22_h[i], P33_h[i], P21_h[i], P31_h[i], P32_h[i], A_h[i]]).astype(np.complex64) )
X_h = np.array(X_h).astype(np.complex64)
K_h = np.empty_like(X_h)
Xs_h = np.empty_like(X_h)
Xm_h = np.empty_like(X_h)

# Allocation of required buffers on the device
X_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=X_h)
K_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=K_h)
Xs_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=Xs_h)
Xm_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=Xm_h)

f = open("source.cl", "r")
source = f.read()
f.close()
prg = cl.Program(ctx, source).build()

print "Begin Calculation"
start_time = time.time()

for t in Timeline:
    print t
    completeevent = prg.RK4Step(queue, (M,), None, X_d, K_d, Xs_d, Xm_d, W, t)
    completeevent.wait()
    
cl.enqueue_copy(queue, X_h, X_d)
end_time = time.time()
print "All done"
print "Calculation took " + str(end_time - start_time) + " seconds"

A_h = X_h[:,6]
plt.plot(np.real(A_h))
plt.plot(np.abs(A_h))
plt.show()
