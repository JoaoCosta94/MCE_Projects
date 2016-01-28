import pyopencl as cl
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
MF = cl.mem_flags

M = 3

zero = np.complex64(0.0)

X1_h = np.array([1 + 1j*2, 2 + 1j*3, 3 + 1j*4]).astype(np.complex64)
X2_h = np.array([1 + 1j*2, 2 + 1j*3, 3 + 1j*4]).astype(np.complex64)
X3_h = np.array([1 + 1j*2, 2 + 1j*3, 3 + 1j*4]).astype(np.complex64)
Y1_h = np.array([4 + 1j*5, 5 + 1j*6, 6 + 1j*7]).astype(np.complex64)
Y2_h = np.array([4 + 1j*5, 5 + 1j*6, 6 + 1j*7]).astype(np.complex64)
Y3_h = np.array([4 + 1j*5, 5 + 1j*6, 6 + 1j*7]).astype(np.complex64)

A_h = np.array([1+1j, 1+1j, 1+1j]).astype(np.complex64)
X_h = np.array([2, 2, 2]).astype(np.float32)

RES_h = np.empty_like(X1_h)

data_h = []
for i in range(M):
      data_h.append(np.array([X1_h[i], X2_h[i], X3_h[i], Y1_h[i], Y2_h[i], Y3_h[i]]).astype(np.complex64))
data_h = np.array(data_h).astype(np.complex64)

A_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=A_h)
data_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=data_h)
X_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=X_h)
RES_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf = RES_h)

Source = """
    #define complex_ctr(x, y) (float2)(x, y)
    #define complex_mul(a, b) complex_ctr(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y))
    #define complex_unit (float2) (0, 1)
    #define complex_exp(a) complex_ctr(cos(a), sin(a))

    __kernel void sum( __global float2 *data,
                       __global float2 *A,
                       __global float *X,
                       __global float2 *res,
                       int rowWidth){
        const int gid_x = get_global_id(0);
        res[gid_x] = A[gid_x] + complex_mul(complex_mul(data[gid_x*rowWidth+3], complex_exp(X[gid_x])), complex_unit);
    }
"""
prg = cl.Program(ctx, Source).build()

print "DATA"
print data_h
print "Numpy Result"
print A_h + 1j*np.exp(1j*X_h)*data_h[:, 3]

sum = prg.sum(queue, (M,), None, data_d, A_d, X_d, RES_d, np.int32(len(data_h[0])))
sum.wait()
cl.enqueue_copy(queue, RES_h, RES_d)

print 'Device Result'
print RES_h