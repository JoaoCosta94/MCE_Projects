__author__ = 'Joao Costa'

import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import numpy.linalg as la
from pyopencl.elementwise import ElementwiseKernel
import matplotlib.pyplot as plt

"""
Solve the problem
Xi' = M1*Xi + M2*Xi~
M1 and M2 are 6*6 Complex Matrixes
Xi in the form [P11i, P22i, P33i, P21i, P31i, P32i] where Pxyi is a complex number
""" 

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
MF = cl.mem_flags

# Constants 
M = 10 # Number of atoms
L = np.float32(0.000000001) # Atom Spacing
N = 1000 # Number of time intervals
dt = np.float32(0.01) # Time interval
Timeline = np.arange(0.0, N, dt) 
P0 = np.float32(1.0)
Delta = np.float32(1.0)
Gama = np.float32(1.0)
Omc = np.float32(1.0)

# acabar os defines dos valores
text = "#define M (int)" + str(M) + "\n"
text += "#define L (float)" + str(L) + "\n"
text += "#define dt (float) " + str(dt) + "\n"
text += "#define P0 (float) " + str(P0) + "\n"
text += "#define Delta (float) " + str(Delta) + "\n"
text += "#define Gama (float) " + str(Gama) + "\n"
text += "#define Omc (float)" + str(Omc) + "\n"
f1 = open("source.cl",'w+')
f2 = open("kernel.cl", "r")
kernel = f2.read()
f1.write(text + kernel)
f2.close()
f1.close()

#Initial Conditions OmegaP is yet missing here 
P11_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P22_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P33_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P21_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P31_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P32_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
omp = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)

f = open("source.cl", "r")
source = f.read()
f.close()
prg = cl.Program(ctx, source).build()

X_h = []
OMP_h = []
for i in range(M):
    X_h.append( np.array( [P11_h[i].real, P11_h[i].imag, P22_h[i].real, P22_h[i].imag, P33_h[i].real, P33_h[i].imag, P21_h[i].real, P21_h[i].imag, P31_h[i].real, P31_h[i].imag, P32_h[i].real, P32_h[i].imag] ) )
    #OMP_h.append( np.array( [omp[i].real, omp[i].imag] ) )
X_h = np.array(X_h)
#OMP_h = np.arrau(OMP_h)

X_d = cl_array.to_device(queue, X_h)
OMP_d = cl_array.to_device(queue, OMP_h)

for t in Timeline:
    completeevent = prg.RK4Step(queue, (M,), None, X_d, OMP_d)
    completeevent.wait()
