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

dados_h = []

for i in range(3):
      #a = np.array([P11_h[i], P22_h[i], P33_h[i], P21_h[i], P31_h[i], P32_h[i]]).astype(np.complex64)
      a = np.array([P11_h[i], P22_h[i]]).astype(np.complex64)
      dados_h.append(a)

dados2_h = dados_h[:]
dados_h = np.array(dados_h).astype(np.complex64)
dados2_h = np.array(dados2_h).astype(np.complex64)
RES_h = np.empty_like(dados_h)

print "dados1"
print dados_h
print "\n Expected"
print dados_h * 2

dados_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=dados_h)
dados2_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=dados2_h)
RES_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf = RES_h)


Source = """
__kernel void soma(__global float2 *dados, __global float2 *dados2, __global float2 *res, int W){
	const int gid_x = get_global_id(0);
	for(int i = 0; i<W; i++)
	{
             res[gid_x*W+i] = dados[gid_x*W+i] *2;
	}
}
"""
prg = cl.Program(ctx, Source).build()

completeEvent = prg.soma(queue, (M,), None, dados_d, dados2_d, RES_d, np.int32(2))
completeEvent.wait()

cl.enqueue_copy(queue, RES_h, RES_d)
print "\n RES"
print RES_h - (dados_h + dados2_h)
