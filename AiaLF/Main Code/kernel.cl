
void f(__global float2 *X,  
	   __global float2 *K, 
	   int id, 
	   uint W,
	   float t){  
	float exp_arg;
	float2 p11, p22, p33, p21, p31, p32, op, aux, ar, al, p;
	p11 = X[id*W];
	p22 = X[id*W+1];
	p33 = X[id*W+2];
	p21 = X[id*W+3];
	p31 = X[id*W+4];
	p32 = X[id*W+5];
	al = X[(id-1)*W+6];
	ar = X[(id+1)*W+6];
	op = p0 * complex_mul(X[id*W+6], complex_unit);
	
	aux = p22 * gama/2 + complex_mul(op, p22) + conj(p22) * gama/2 + complex_mul(op, conj(p22));
	K[id*W] = aux;
	
	aux = (-p22*gama - complex_mul(op, p21) + complex_mul(p32, complex_unit)*omc 
	      - conj(p22)*gama - complex_mul(op, conj(p21)) + complex_mul(conj(p32), complex_unit)*omc);
	K[id*W+1] = aux;
	
	aux = p22*gama/2 - complex_mul(p32, complex_unit)*omc + conj(p22)*gama/2 - complex_mul(conj(p32), complex_unit)*omc;
	K[id*W+2] = aux;
	
	aux = complex_mul(op, p11) - complex_mul(op, p22) - p21*gama + complex_mul(p21, complex_unit)*delta + complex_mul(p31, complex_unit)*omc;	 
	K[id*W+3] = aux;
	
	aux = complex_mul(p21, complex_unit)*omc + complex_mul(p31, complex_unit)*delta - complex_mul(op, p32);
	K[id*W+4] = aux;
	
	aux = (complex_mul(p22, complex_unit)*omc - complex_mul(p33, complex_unit)*omc - complex_mul(op, p31) - p32*gama);
	K[id*W+5] = aux;
	
	exp_arg = k_p * L * id - om_p * t;
	p = complex_mul(b*p0*p21*complex_ctr(cos(exp_arg), sin(exp_arg)), complex_unit);
	//aux = complex_mul(v *(ar + al), complex_unit);
	aux = aux + p;
	K[id*W+6] = aux;
}

__kernel void RK4Step(__global float2 *X, 
					  __global float2 *K, 
					  __global float2 *Xs, 
					  __global float2 *Xm, 
					  uint W,
					  float t){
    const int gid_x = get_global_id(0);
	int idx = 0;	

    //computation of k1
    f(X, K, gid_x, W, t);
	for(int i=0; i<W; i++)
	{
		idx = gid_x*W+i;
		Xs[idx] = X[idx] + dt*K[idx]/6;
		Xm[idx] = X[idx] + dt*K[idx]/2;
	}
    
    //computation of k2
    f(Xm, K, gid_x, W, t);
	for(int i=0; i<W; i++)
	{
		idx = gid_x*W+i;
		Xs[idx] = Xs[idx] + dt*K[idx]/3;
		Xm[idx] = X[idx] + dt*K[idx]/2;
	}	

    //computation of k3
    f(Xm, K, gid_x, W, t);
	for(int i=0; i<W; i++)
	{
		idx = gid_x*W+i;
		Xs[idx] = Xs[idx] + dt*K[idx]/3;
		Xm[idx] = X[idx] + dt*K[idx];
	}	

    //computation of k4
    f(Xm, K, gid_x, W, t);
	for(int i=0; i<W; i++)
	{
		idx = gid_x*W+i;
		Xs[idx] = Xs[idx] + dt*K[idx]/6;
	}

    //update photon
	for(int i=0; i<W; i++)
	{
		idx = gid_x*W+i;
		X[idx] = Xs[idx];
	}
}