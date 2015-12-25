from __future__ import absolute_import
from __future__ import print_function
import pyopencl as cl
import pyopencl.array as cl_array
import numpy
import numpy.linalg as la

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

n = 100

x1_gpu = cl_array.to_device(queue,
        ( numpy.random.randn(n) + 1j*numpy.random.randn(n)
            ).astype(numpy.complex64))
x2_gpu = cl_array.to_device(queue,
        ( numpy.random.randn(n) + 1j*numpy.random.randn(n)
            ).astype(numpy.complex64))
b_gpu = cl_array.to_device(queue,
        ( numpy.random.randn(n) + 1j*numpy.random.randn(n)
            ).astype(numpy.complex64))
f1_gpu = cl_array.to_device(queue,
        ( numpy.random.randn(n) + 1j*numpy.random.randn(n)
            ).astype(numpy.complex64))
f2_gpu = cl_array.to_device(queue,
        ( numpy.random.randn(n) + 1j*numpy.random.randn(n)
            ).astype(numpy.complex64))
a = 100.0
dt = 0.001
            
precode =  """
        #define complex_ctr(x, y) (float2)(x, y)
        #define complex_add(a, b) complex_ctr((a).x + (b).x, (a).y + (b).y)
        #define complex_mul(a, b) complex_ctr(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y))
        #define complex_mul_scalar(a, b) complex_ctr((a).x * (b), (a).y * (b))
        #define complex_div_scalar(a, b) complex_ctr((a).x / (b), (a).y / (b))
        #define conj(a) complex_ctr((a).x, -(a).y)
        #define conj_transp(a) complex_ctr(-(a).y, (a).x)
        #define conj_transp_and_mul(a, b) complex_ctr(-(a).y * (b), (a).x * (b))
        """
         
from pyopencl.elementwise import ElementwiseKernel
df = ElementwiseKernel(ctx,
        "float a, "
        "float2 *b, "
        "float2 *x1, "
        "float2 *x2, "
        "float2 *f1, "
        "float2 *f2 ",
        """
        f1[i] = complex_mul_scalar(x1[i], a) + complex_mul(b[i],x2[i]) + complex_mul(b[i],x2[i+1]);
        f2[i] = complex_mul_scalar(x2[i], -a) + complex_mul(b[i],x1[i])
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

complex_add = ElementwiseKernel(ctx,
        "float2 *x, "
        "float2 *y, "
        "float2 *z",
        "z[i] = x[i] + y[i]",
        "complex_add")

real_part = ElementwiseKernel(ctx,
        "float2 *x, float *z",
        "z[i] = x[i].x",
        "real_part")
import matplotlib.pyplot as plt
x1_gpu_real = cl_array.empty(queue, len(x1_gpu), dtype=numpy.float32)
for i in range(1000):
    b_gpu = cl_array.to_device(queue,
        ( numpy.random.randn(n) + 1j*numpy.random.randn(n)
            ).astype(numpy.complex64))
    df(a, b_gpu, x1_gpu, x2_gpu, f1_gpu, f2_gpu)
    Euler(dt, a, b_gpu, x1_gpu, x2_gpu, f1_gpu, f2_gpu)
    real_part(x1_gpu, x1_gpu_real)
    plt.plot(x1_gpu_real.get())
plt.show()