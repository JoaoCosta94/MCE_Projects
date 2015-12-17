#define M (int)10
#define L (float)1e-09
#define dt (float) 0.01
#define P0 (float) 1.0
#define Delta (float) 1.0
#define Gama (float) 1.0
#define Omc (float)1.0
#define complex_ctr(x, y) (float2)(x, y)
#define complex_add(a, b) complex_ctr((a).x + (b).x, (a).y + (b).y)
#define complex_mul(a, b) complex_ctr(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y))
#define complex_mul_scalar(a, b) complex_ctr((a).x * (b), (a).y * (b))
#define complex_div_scalar(a, b) complex_ctr((a).x / (b), (a).y / (b))
#define conj(a) complex_ctr((a).x, -(a).y)
#define conj_transp(a) complex_ctr(-(a).y, (a).x)
#define conj_transp_and_mul(a, b) complex_ctr(-(a).y * (b), (a).x * (b))
#define complex_real(a) a.x
#define complex_imag(a) a.y
#define complex_unit (float2)(0, 1)

float16 f(float16 x, float2 omp)
{ 
    float16 v;
	float2 p11, p22, p33, p21, p31, p32, aux;
	p11 = complex_ctr(x.s0, x.s1);
	p22 = complex_ctr(x.s2, x.s3);
	p33 = complex_ctr(x.s4, x.s5);
	p21 = complex_ctr(x.s6, x.s7);
	p31 = complex_ctr(x.s8, x.s9);
	p32 = complex_ctr(x.s10, x.s11);
	omp = complex_mul(omp, complex_unit);
	
	aux = complex_mul_scalar(p22, Gama/2.0) + complex_mul(omp, p22) + complex_mul_scalar(conj(p22), Gama/2.0) + complex_mul(omp, conj(p22));
	v.s0 = complex_real(aux);
	v.s1 = complex_imag(aux);
	
	aux = (complex_mul_scalar(p22, -Gama) - complex_mul(omp, p21) + complex_mul_scalar(complex_mul(p32, complex_unit), Omc) 
		   + complex_mul_scalar(conj(p22), -Gama) - complex_mul(omp, conj(p21)) + complex_mul_scalar( complex_mul(conj(p32), complex_unit), Omc));
	v.s2 = complex_real(aux);
	v.s3 = complex_imag(aux);
	
	aux = (complex_mul_scalar(p22, Gama/2.0) - complex_mul_scalar(complex_mul(p32, complex_unit),Omc) 
		  + complex_mul_scalar(conj(p22), Gama/2.0) - complex_mul_scalar(complex_mul(conj(p32), complex_unit)), Omc);
	v.s4 = complex_real(aux);
	v.s5 = complex_imag(aux);
	
	aux = (complex_mul(omp, p11) - complex_mul(omp, p22) + complex_mul_scalar(p21, -Gama) 
	      + complex_mul_scalar(complex_mul(p21, complex_unit), Delta) + complex_mul_scalar(complex_mul(p31, complex_unit), Omc));	 
	v.s6 = complex_real(aux);
	v.s7 = complex_imag(aux);
	
	aux = complex_mul_scalar(complex_mul(p21, complex_unit), Omc) + complex_mul_scalar(complex_mul(p31, complex_unit), Delta) - complex_mul(omp, p32);
	v.s8 = complex_real(aux);
	v.s9 = complex_imag(aux);
	
	aux = (complex_mul_scalar(complex_mul(p22, complex_unit), Omc) - complex_mul_scalar(complex_mul(p33, complex_unit), Omc)
		   -complex_mul(omp, p31) +  complex_mul_scalar(p32, -Gama));
	v.s10 = complex_real(aux);
	v.s11 = complex_imag(aux);
    return v;
}

__kernel void RK4Step(__global float16 *x, __global float2 *omp){
    const int gid = get_global_id(0);
    float16 k, xm,xs;


    //k1
    k = f(x[gid], omp[gid]);
    xs = x[gid] + dt * k/6.0;
    xm = x[gid] + 0.5 * dt * k;

    //k2
    k = f(xm, omp[gid]);
    xs +=  dt * k/3.0;
    xm = x[gid] + 0.5 * dt * k;

    //k3
    k = f(xm, omp[gid]);
    xs +=  dt * k/3.0;
    xm = x[gid] + dt * k;

    //k4
    k = f(xm, omp[gid]);
    xs +=  dt * k/6.0;

    //update photon
    x[gid] = xs;
}