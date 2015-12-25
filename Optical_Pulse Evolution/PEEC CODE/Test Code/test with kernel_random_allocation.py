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
      a = np.array([P11_h[i], P22_h[i]]).astype(np.complex64)
      dados_h.append(a)

dados_h = np.array(dados_h).astype(np.complex64)
k_h = np.empty_like(dados_h)
RES_h = np.empty_like(dados_h)

print "dados1"
print dados_h

dados_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=dados_h)
k_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=k_h)
RES_d = cl.Buffer(ctx, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf = RES_h)


Source = """
void f(__global float2 *dados, __global float2 *k, int id, int W)
{ 
    float2 q0 = dados[id*W];
    float2 q1 = dados[id*W + 1];
    k[id*W] = q0;
    k[id*W+1] = q1;
}

__kernel void soma(__global float2 *dados, __global float2 *k, __global float2 *res, int W){
	const int gid_x = get_global_id(0);
	f(dados, k , gid_x, W);
	for(int i=0; i<W; i++)
	{
            res[gid_x*W+i] = k[gid_x*W+i]*2;
	}
}
"""
prg = cl.Program(ctx, Source).build()

completeEvent = prg.soma(queue, (M,), None, dados_d, k_d, RES_d, np.int32(2))
completeEvent.wait()

cl.enqueue_copy(queue, RES_h, RES_d)
cl.enqueue_copy(queue, k_h, k_d)
print "\n dados - k"
print dados_h - k_h
print "\n Expected"
print dados_h*2
print "\n RES"
print RES_h - dados_h*2
