
void evolve_state(__global float2 *P,
                  __global float2 *A,
                  __global float *OC,
                  __global float2 *K,
                  int id,
                  uint W){
	//p11 = P[id*W];
	//p22 = P[id*W+1];
	//p33 = P[id*W+2];
	//p21 = P[id*W+3];
	//p31 = P[id*W+4];
	//p32 = P[id*W+5];

	K[id*W] = (GAMA*(P[id*W+1] + conj(P[id*W+1]))/2
	           + complex_mul(P0*complex_mul(A[id], (P[id*W+3] + conj(P[id*W+3]))), complex_unit));

	K[id*W+1] = (-GAMA*(P[id*W+1] + conj(P[id*W+1])) - complex_mul(P0*complex_mul(A[id], (P[id*W+3] + conj(P[id*W+3])))
				 - OC[id]*(P[id*W+5] + conj(P[id*W+5])), complex_unit));

	K[id*W+2] = (GAMA*(P[id*W+1] + conj(P[id*W+1]))/2
	             - complex_mul(OC[id]*(P[id*W+5] + conj(P[id*W+5])), complex_unit));

	K[id*W+3] = (complex_mul(P0*complex_mul(A[id],(P[id*W] - P[id*W+1])) + OC[id]*P[id*W+4]
	             - DELTA*P[id*W+3], complex_unit) - GAMA*P[id*W+3]);

	K[id*W+4] = complex_mul(-P0*complex_mul(A[id], P[id*W+5]) + OC[id]*P[id*W+3] + DELTA*P[id*W+4], complex_unit);

	K[id*W+5] = (complex_mul(-P0*complex_mul(A[id], P[id*W+4]) + OC[id]*(P[id*W+1] - P[id*W+2]), complex_unit)
	             - GAMA*P[id*W+5]);
}

__kernel void RK4Step(__global float2 *P,
 					  __global float2 *A,
 					  __global float *OC,
					  __global float2 *K, 
					  __global float2 *Ps,
					  __global float2 *Pm,
					  uint W,
					  float t){
    const int gID_x = get_global_id(0);
	int idx = 0;	

    //computation of k1
    evolve_state(P, A, OC, K, gID_x, W);
	for(int i=0; i<W; i++)
	{
		idx = gID_x*W+i;
		Ps[idx] = P[idx] + dt*K[idx]/6;
		Pm[idx] = P[idx] + dt*K[idx]/2;
	}
    
    //computation of k2
    evolve_state(Pm, A, OC, K, gID_x, W);
	for(int i=0; i<W; i++)
	{
		idx = gID_x*W+i;
		Ps[idx] = Ps[idx] + dt*K[idx]/3;
		Pm[idx] = P[idx] + dt*K[idx]/2;
	}	

    //computation of k3
    evolve_state(Pm, A, OC, K, gID_x, W);
	for(int i=0; i<W; i++)
	{
		idx = gID_x*W+i;
		Ps[idx] = Ps[idx] + dt*K[idx]/3;
		Pm[idx] = P[idx] + dt*K[idx];
	}	

    //computation of k4
    evolve_state(Pm, A, OC, K, gID_x, W);
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