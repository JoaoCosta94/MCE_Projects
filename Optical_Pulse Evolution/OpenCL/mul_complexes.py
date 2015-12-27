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
aux_h = np.array([1+1j, 1+1j, 1+1j, 1+1j, 1+1j, 1+1j]).astype(np.complex64)
aux2_h = np.array([2, 2, 2, 2, 2, 2]).astype(np.float32)
RES_h = np.empty_like(X1_h)

dados_h = []
for i in range(3):
      dados_h.append(np.array([X1_h[i], X2_h[i], X3_h[i], Y1_h[i], Y2_h[i], Y3_h[i]]).astype(np.complex64))
dados_h = np.array(dados_h).astype(np.complex64)

print dados_h

aux_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=aux_h)
aux2_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=aux2_h)
dados_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=dados_h)
RES_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf = RES_h)

Source = """
	#define complex_ctr(x, y) (float2)(x, y)
	#define complex_mul(a, b) complex_ctr(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y))

	__kernel void soma( __global float2 *dados, __global float2 *aux, __global float *aux2, __global float2 *res, int rowWidth){
		const int gid_x = get_global_id(0);
		res[gid_x] = aux2[gid_x]*complex_mul(dados[gid_x*rowWidth+3], aux[gid_x]);
	}
"""
prg = cl.Program(ctx, Source).build()

soma = prg.soma(queue, (M,), None, dados_d, aux_d, aux2_d, RES_d, np.int32(6))
soma.wait()

cl.enqueue_copy(queue, RES_h, RES_d)
print "GPU RES"
print RES_h

