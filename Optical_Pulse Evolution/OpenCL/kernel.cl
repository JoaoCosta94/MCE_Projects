
void evolve_state(__global float2 *P,
                  __global float2 *A,
                  __global float2 *K,
                  int id,
                  uint W){
	float2 p11, p22, p33, p21, p31, p32;
	p11 = P[id*W];
	p22 = P[id*W+1];
	p33 = P[id*W+2];
	p21 = P[id*W+3];
	p31 = P[id*W+4];
	p32 = P[id*W+5];

	K[id*W] = GAMA*(p22 + conj(p22))/2 + complex_mul(P0*A[id]*(p21 + conj(p21)), complex_unit);

	K[id*W+1] = -GAMA*(p22 + conj(p22)) - complex_mul(P0*A[id]*(p21 + conj(p21)) - OC*(p32 + conj(p32)), complex_unit);

	K[id*W+2] = GAMA*(p22 + conj(p22))/2 - complex_mul(OC*(p32 + conj(p32)), complex_unit);

	K[id*W+3] = complex_mul(P0*A[id]*(p11 - p22) + OC*p31 - DELTA*p21, complex_unit) - GAMA*p21;

	K[id*W+4] = complex_mul(-P0*A[id]*p32 + OC*p21 + DELTA*p31, complex_unit);

	K[id*W+5] = complex_mul(-P0*A[id]*p31 + OC*(p22 - p33), complex_unit) - GAMA*p32;
}

__kernel void RK4Step(__global float2 *P,
 					  __global float2 *A,
					  __global float2 *K, 
					  __global float2 *Ps,
					  __global float2 *Pm,
					  uint W,
					  float t){
    const int gID_x = get_global_id(0);
	int idx = 0;	

    //computation of k1
    evolve_state(P, A, K, gID_x, W);
	for(int i=0; i<W; i++)
	{
		idx = gID_x*W+i;
		Ps[idx] = P[idx] + dt*K[idx]/6;
		Pm[idx] = P[idx] + dt*K[idx]/2;
	}
    
    //computation of k2
    evolve_state(Pm, A, K, gID_x, W);
	for(int i=0; i<W; i++)
	{
		idx = gID_x*W+i;
		Ps[idx] = Ps[idx] + dt*K[idx]/3;
		Pm[idx] = P[idx] + dt*K[idx]/2;
	}	

    //computation of k3
    evolve_state(Pm, A, K, gID_x, W);
	for(int i=0; i<W; i++)
	{
		idx = gID_x*W+i;
		Ps[idx] = Ps[idx] + dt*K[idx]/3;
		Pm[idx] = P[idx] + dt*K[idx];
	}	

    //computation of k4
    evolve_state(Pm, A, K, gID_x, W);
	for(int i=0; i<W; i++)
	{
		idx = gID_x*W+i;
		Ps[idx] = Ps[idx] + dt*K[idx]/6;
	}

    //update state
	for(int i=0; i<W; i++)
	{
		idx = gID_x*W+i;
		P[idx] = Ps[idx];
	}
}