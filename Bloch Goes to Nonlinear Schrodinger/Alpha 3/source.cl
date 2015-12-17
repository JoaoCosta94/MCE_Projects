#define M 10
#define L 1e-09
#define dt 0.01
#define P0 1.0
#define Delta 1.0
#define Gama 1.0
#define Omc 1.0
#define complex_ctr(x, y) (float2)(x, y)
#define complex_add(a, b) complex_ctr((a).x + (b).x, (a).y + (b).y)
#define complex_mul(a, b) complex_ctr(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y))
#define complex_mul_scalar(a, b) complex_ctr((a).x * (b), (a).y * (b))
#define complex_div_scalar(a, b) complex_ctr((a).x / (b), (a).y / (b))
#define conj(a) complex_ctr((a).x, -(a).y)
#define conj_transp(a) complex_ctr(-(a).y, (a).x)
#define conj_transp_and_mul(a, b) complex_ctr(-(a).y * (b), (a).x * (b))
#define complex_unit (float2)(0, 1)

float8 f(float s,  float8 q, float omp)
{ 
    float8 v;
	
	omp = complex_mul(omp, complex_unit)
	
	v.s0 = complex_mul_scalar(q.s1, gama/2.0) + complex_mul(q.s3, omp) + complex_mul_scalar(complex_transp(q.s1), gama/2.0) + complex_mul(complex_transp(q.s3), omp);
	v.s1 = q.s5;
	v.s2 = q.s6;
	v.s3 = q.s7;
	
	v.s4 = -q.s0;
	v.s5 = -q.s1;
	v.s6 = -q.s2;
	v.s7 = -q.s3;

    return v;
}

__kernel void RK4Step(float s, float dt, __global float8 *q, __global float *omp){
    const int gid = get_global_id(0);
    float8 k, qm,qs;


    //k1
    k = f(s, q[gid], omp[gid]);
    qs = q[gid] + dt * k/6.0;
    qm = q[gid] + 0.5 * dt * k;

    //k2
    k = f(s+0.5*dt, qm, omp[gid]);
    qs +=  dt * k/3.0;
    qm = q[gid] + 0.5 * dt * k;

    //k3
    k = f(s+0.5*dt,  qm, omp[gid]);
    qs +=  dt * k/3.0;
    qm = q[gid] + dt * k;

    //k4
    k = f(s + dt, qm, omp[gid]);
    qs +=  dt * k/6.0;

    //update photon
    q[gid] = qs;
}