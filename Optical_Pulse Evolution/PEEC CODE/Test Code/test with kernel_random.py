import pyopencl as cl
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
MF = cl.mem_flags

M = 3

zero = np.complex64(0.0)
P11_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P22_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P33_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P21_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P31_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
P32_h = (np.random.randn(M) + 1j*np.random.randn(M)).astype(np.complex64)
aux_h = np.complex64(1 + 1j*1)
RES_h = np.empty_like(P11_h)

dados_h = []
for i in range(3):
      a = np.array([P11_h[i], P22_h[i], P33_h[i], P21_h[i], P31_h[i], P32_h[i]]).astype(np.complex64)
      dados_h.append(a)
dados_h = np.array(dados_h).astype(np.complex64)

print dados_h

aux_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=aux_h)
dados_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=dados_h)
RES_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf = RES_h)


Source = """
__kernel void soma(__global float2 *aux, __global float2 *dados, __global float2 *res, int W){
	const int gid_x = get_global_id(0);
	res[gid_x] = dados[gid_x*W+1] * 2;
}
"""
prg = cl.Program(ctx, Source).build()

completeEvent = prg.soma(queue, (M,), None, aux_d, dados_d, RES_d, np.int32(6))
completeEvent.wait()

cl.enqueue_copy(queue, RES_h, RES_d)
print "GPU RES"
print RES_h
