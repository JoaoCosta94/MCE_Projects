
import scipy as sp
import pyopencl as cl
import pylab as pl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
MF = cl.mem_flags

dx = 0.01
X = sp.arange(0.0, 100, dx)
N = len(X);

A_h = (sp.exp(-(X-50)**2/10.)*sp.exp(-1j*1000*X)).astype(sp.complex64)
A_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=A_h)

pl.figure()
pl.plot(X, abs(A_h)**2/(abs(A_h)**2).max())

Source = """
    #define complex_ctr(x, y) (float2)(x, y)
    #define complex_mul(a, b) complex_ctr(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y))
    #define complex_unit (float2)(0, 1)

    __kernel void propagate(__global float2 *A){
        const int gid_x = get_global_id(0);
        float EPS = 0.1;
        A[gid_x] = A[gid_x] + EPS*complex_mul((A[gid_x+1] + A[gid_x-1]), complex_unit);
    }
"""
prg = cl.Program(ctx, Source).build()
for i in range(1000):
    print i
    event = prg.propagate(queue, (N,), None, A_d)
    event.wait()
cl.enqueue_copy(queue, A_h, A_d)

print "olha o plot"
pl.plot(X, abs(A_h)**2/(abs(A_h)**2).max())

pl.show()