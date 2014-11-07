#include <cstdlib>
#include <cuda_runtime.h>
#include "helper_math.h"	// overload operators for floatN
#include "helper_cuda.h"
#include "random.cuh"

typedef struct 
{
	float3 X;	// position
	float3 P;	// linear impulse
    float radius;
    float mass;
} ParticleState;

typedef struct
{
    float3 position;
    float3 velocity;
    float radius;
    float inv_mass;
} CollisionObject;

#define delta_spring 8000
#define delta_dashpot 30
#define delta_shear 10
#define delta_reflect 0.6

#define BOX_SIZE 3.5


// integrators

//! euler1
//
//
__device__ ParticleState euler(ParticleState curState,
								float3 F,
								float w,
								float dT)
{
	float3 curV = w*curState.P;
	
	ParticleState result;
	result.X = curState.X + dT*curV;
	result.P = curState.P + dT*F;

	return result;
}

//! rungeKutta4XP
// - integrates an old position via rk4 regarding position and impulse
//
__device__ ParticleState rungeKutta4XP(ParticleState curState,
										float3 F,		// current linear force
										float w,		// inverse mass of particle
										float dT)		// step in time
{
	// time precalc
	float halfdt = dT * 0.5f;
	float sixthdt = dT / 6.0f;

	// cache
	float3 newV;

	// get current velocity
	float3 curV = w * curState.P; 
	
	// 1
	float3 A1_X_DT = curV;
	float3 A1_P_DT = F;
	newV = w * (curState.P + halfdt * A1_P_DT);
		
	//2
	float3 A2_X_DT = newV;
	float3 A2_P_DT = F;
	newV = w * (curState.P + halfdt * A2_P_DT);
	
	//3
	float3 A3_X_DT = newV;
	float3 A3_P_DT = F;
	newV = w * (curState.P + dT * A3_P_DT);
	
	// 4
	float3 A4_X_DT = newV;
	float3 A4_P_DT = F;
	
	// final update
	ParticleState result;
	result.X = curState.X + sixthdt * (A1_X_DT + 2.0f * (A2_X_DT + A3_X_DT) + A4_X_DT);
	result.P = curState.P + sixthdt * (A1_P_DT + 2.0f * (A2_P_DT + A3_P_DT) + A4_P_DT);
		
	return result;
}

__device__ float3 springDashpotForce(CollisionObject a, CollisionObject b) {
    float3 ab = b.position - a.position;
    float len_ab = length(ab);
    float3 n_ab = ab / len_ab;
    float3 v_rel = b.velocity - a.velocity;

    float3 f_spring = delta_spring * (a.radius + b.radius - len_ab) * n_ab;
    float3 f_dashpot = -delta_dashpot * v_rel;
    float3 f_shear = delta_shear * (v_rel - dot(v_rel, -n_ab) * (-n_ab));
    float3 f_total = f_spring + f_dashpot + f_shear;

    return f_total;
}

__device__ void respondSphereCollisions(float* ptVbo, float* f_tmp, int i, int j) {
    CollisionObject a;
    a.position = make_float3(ptVbo[12*i+0], ptVbo[12*i+1], ptVbo[12*i+2]);
    a.radius = ptVbo[12*i+9];
    a.inv_mass = 1.0f / ptVbo[12*i+11];
    a.velocity = make_float3(ptVbo[12*i+3], ptVbo[12*i+4], ptVbo[12*i+5]) * a.inv_mass;

    CollisionObject b;
    b.position = make_float3(ptVbo[12*j+0], ptVbo[12*j+1], ptVbo[12*j+2]);
    b.radius = ptVbo[12*j+9];
    b.inv_mass = 1.0f / ptVbo[12*j+11];
    b.velocity = make_float3(ptVbo[12*j+3], ptVbo[12*j+4], ptVbo[12*j+5]) * b.inv_mass;

    float3 f = springDashpotForce(a, b);

    atomicAdd(&(f_tmp[3*i+0]), -f.x);
    atomicAdd(&(f_tmp[3*i+1]), -f.y);
    atomicAdd(&(f_tmp[3*i+2]), -f.z);
    atomicAdd(&(f_tmp[3*j+0]), f.x);
    atomicAdd(&(f_tmp[3*j+1]), f.y);
    atomicAdd(&(f_tmp[3*j+2]), f.z);
}

__device__ void detectSphereCollisions(float *ptVbo, float* f_tmp, int numParticles, int i) {
    for (int j = i + 1; j < numParticles; ++j) {
        float3 Xi = make_float3(ptVbo[12*i+0], ptVbo[12*i+1], ptVbo[12*i+2]);
        float3 Xj = make_float3(ptVbo[12*j+0], ptVbo[12*j+1], ptVbo[12*j+2]);
        float ri = ptVbo[12*i+9];
        float rj = ptVbo[12*j+9];
        float3 toJ = Xj - Xi;
        float dist2 = toJ.x * toJ.x + toJ.y * toJ.y + toJ.z * toJ.z;
        if (dist2 < (rj + ri) * (rj + ri)) {
            respondSphereCollisions(ptVbo, f_tmp, i, j);
        }
    }
}

__device__ void detectPlaneCollisions(float *ptVbo, float* f_tmp, int i) {
    float3 position = make_float3(ptVbo[12*i+0], ptVbo[12*i+1], ptVbo[12*i+2]);
    float3 impulse = make_float3(ptVbo[12*i+3], ptVbo[12*i+4], ptVbo[12*i+5]);
    float radius = ptVbo[12*i+9];

    float epsilon = 0.00001;
    if (position.y - radius < 0 && impulse.y < 0) {
        ptVbo[12*i+1] = radius + epsilon;
        ptVbo[12*i+4] *= -delta_reflect;
    }
    if (position.x - radius < -BOX_SIZE/2 && impulse.x < 0) {
        ptVbo[12*i+0] = -BOX_SIZE/2 + radius + epsilon;
        ptVbo[12*i+3] *= -delta_reflect;
    }
    if (position.x + radius > BOX_SIZE/2 && impulse.x > 0) {
        ptVbo[12*i+0] = BOX_SIZE/2 - radius - epsilon;
        ptVbo[12*i+3] *= -delta_reflect;
    }
    if (position.z - radius < -BOX_SIZE/2 && impulse.z < 0) {
        ptVbo[12*i+2] = -BOX_SIZE/2 + radius + epsilon;
        ptVbo[12*i+5] *= -delta_reflect;
    }
    if (position.z + radius > BOX_SIZE/2 && impulse.z > 0) {
        ptVbo[12*i+2] = BOX_SIZE/2 - radius - epsilon;
        ptVbo[12*i+5] *= -delta_reflect;
    }
}

__device__ unsigned int id() {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    seed = i;
    return i;
}


__global__ void box_collisions(float *ptVbo, float *f_tmp, int numParticles)
{
    unsigned int i = id();
    if(i >= numParticles) return;

    // reset force buffer
    f_tmp[3*i + 0] = 0;
    f_tmp[3*i + 1] = 0;
    f_tmp[3*i + 2] = 0;

    detectPlaneCollisions(ptVbo, f_tmp, i);
}

__global__ void sphere_collisions(float* ptVbo, float* f_tmp, int numParticles) {
    unsigned int i = id();
    if(i >= numParticles) return;

    detectSphereCollisions(ptVbo, f_tmp, numParticles, i);
}

__global__ void integrate(float* ptVbo, float* f_tmp, int numParticles, float dT) {
    unsigned int i = id();
    if(i >= numParticles) return;

    // some subtle random velocity changes
    // currentState.P += 0.0000001 * noise3D();

    // init current state
    ParticleState currentState;
    currentState.X.x = ptVbo[12*i + 0];
    currentState.X.y = ptVbo[12*i + 1];
    currentState.X.z = ptVbo[12*i + 2];
    currentState.P.x = ptVbo[12*i + 3];
    currentState.P.y = ptVbo[12*i + 4];
    currentState.P.z = ptVbo[12*i + 5];
    currentState.radius = ptVbo[12*i + 9];
    currentState.mass = ptVbo[12*i + 11];

    // retrieve collision changes
    float3 F = make_float3(f_tmp[3*i + 0], f_tmp[3*i + 1], f_tmp[3*i + 2]);

    // add some gravity
    F += make_float3(0, -10 * currentState.mass, 0);

    // call the intergrator
    ParticleState newState = rungeKutta4XP(currentState, F, 1.f/currentState.mass, dT);

    // save the changes to the buffer
    ptVbo[12*i + 0] = newState.X.x;
    ptVbo[12*i + 1] = newState.X.y;
    ptVbo[12*i + 2] = newState.X.z;
    ptVbo[12*i + 3] = newState.P.x;
    ptVbo[12*i + 4] = newState.P.y;
    ptVbo[12*i + 5] = newState.P.z;
}

// host sided interface code
extern "C" 
{
    void launchParticleKernel(float *ptVbo, float *f_tmp, int numParticles, float t, float dT)
	{
        box_collisions<<<ceil(numParticles/(float)32), 32>>>(ptVbo, f_tmp, numParticles);
        sphere_collisions<<<ceil(numParticles/(float)32), 32>>>(ptVbo, f_tmp, numParticles);
        integrate<<<ceil(numParticles/(float)32), 32>>>(ptVbo, f_tmp, numParticles, dT);
	}
}
