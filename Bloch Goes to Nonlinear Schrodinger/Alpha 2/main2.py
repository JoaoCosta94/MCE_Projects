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
L = 1E-9 # Atom Spacing
N = 1000 # Number of time intervals
dt = np.float32(0.01) # Time interval
Timeline = np.arange(0.0, N, dt) 
Po = np.float32(1.0)
Delta = np.float32(1.0)
Gama = np.float32(1.0)
Omc = np.float32(1.0)

# acabar os defines dos valores
text = "#define M " 
f = open("constants.cl",'w+')
f.write(str(M))

#Initial Conditions OmegaP is yet missing here 
P11_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P22_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P33_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P21_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P31_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P32_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
Omp_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)

P11_d = cl_array.to_device(queue, P11_h)
P22_d = cl_array.to_device(queue, P22_h)
P33_d = cl_array.to_device(queue, P33_h)
P21_d = cl_array.to_device(queue, P21_h)
P31_d = cl_array.to_device(queue, P31_h)
P32_d = cl_array.to_device(queue, P32_h)
Omp_d = cl_array.to_device(queue, Omp_h)

f = open("precode.cl", "r")
precode = f.read()

df = ElementwiseKernel(ctx,
        "float2 *p11, "
        "float2 *p22, "
        "float2 *p33, "
        "float2 *p21, "
        "float2 *p31, "
        "float2 *p32 ",
        """
        f1[i] = complex_mul_scalar(x1[i], a) + complex_mul(b[i],x2[i]) + complex_mul(b[i],x2[i+1]);
        f2[i] = complex_mul_scalar(x2[i], -a) + complex_mul(b[i],x1[i]);
        f3[i] = complex_mul_scalar(x2[i], -a) + complex_mul(b[i],x1[i]);
        f4[i] = complex_mul_scalar(x2[i], -a) + complex_mul(b[i],x1[i]);
        f5[i] = complex_mul_scalar(x2[i], -a) + complex_mul(b[i],x1[i]);
        f6[i] = complex_mul_scalar(x2[i], -a) + complex_mul(b[i],x1[i])
        """,
        "df",
        preamble=precode)

Euler = ElementwiseKernel(ctx,
        "float dt, "
        "float a, "
        "float2 *b, "
        "float2 *x1, "
        "float2 *x2, "
        "float2 *f1, "
        "float2 *f2 ",
        """
        x1[i] = x1[i] + complex_mul_scalar(f1[i], dt);
        x2[i] = x2[i] + complex_mul_scalar(f2[i], dt)
        """,
        "Euler",
        preamble=precode)

for t in Timeline:
    t = np.float32(t)
    completeEvent = prg.RK4Step(queue, (M,), None, t, X_d, Omp_d)
    completeEvent.wait()
