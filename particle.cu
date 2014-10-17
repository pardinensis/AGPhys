#include <cstdlib>
#include <cuda_runtime.h>
#include "helper_math.h"	// overload operators for floatN
#include "helper_cuda.h"

typedef struct 
{
	float3 X;	// position
	float3 P;	// linear impulse
	float age;  // remaining time to live
} ParticleState;

// noise framework
__device__ uint seed;

__device__ float noise()
{
	seed = (seed * 1103515245u + 12345u);
	return float(seed)/4294967296.0f;
}

__device__ float2 noise2D()
{
	float2 result;
	seed = (seed * 1103515245u + 12345u);	result.x = float(seed);
	seed = (seed * 1103515245u + 12345u);	result.y = float(seed);
	return (result/2147483648.0f)-make_float2(1.0f,1.0f);
}

__device__ float3 noise3D()
{
	float3 result;
	seed = (seed * 1103515245u + 12345u);	result.x = float(seed);
	seed = (seed * 1103515245u + 12345u);	result.y = float(seed);
	seed = (seed * 1103515245u + 12345u);	result.z = float(seed);
	return (result/2147483648.0f)-make_float3(1.0f,1.0f,1.0f);
}

__device__ float3 distractDirection3D(float3 vIn, float range)
{
	float2 N = noise2D();
	float phi = 2.0f * 3.1415926536f * N.y;
	float cosTheta = sqrtf(1.0f-N.x);
	float sinTheta = sqrtf(1.0f-cosTheta*cosTheta);
	float3 localSampling = make_float3(sinTheta*cosf(phi), cosTheta, sinTheta*sinf(phi));
	localSampling.x *= range;
	localSampling.z *= range;

	// transform to worldspace
	float3 y = vIn;
	float3 h = y;
	if(fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z))
	{
		h.x = 1.0f;
	}
	else if(fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z))
	{
		h.y = 1.0f;
	}
	else
	{
		h.z = 1.0f;
	}

	float3 x = normalize(cross(h,y));
	float3 z = normalize(cross(x,y));

	return normalize(localSampling.x * x + localSampling.y * y + localSampling.z * z);
}

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

// color helper
__device__ float4 colorForAge(float age)
{
	float4 result;
	result.x = (30.0f+age)/80.0f;
	result.y = (5.0f+age)/55.0f;
	result.z = (age*age/2500.0f);
	result.w = 1.0f;

	return result;
}

//! particleKernel - gpu code
// - outPos: float4 array of particle positions
// - time: delta time until last frame
__global__ void particleKernel(float *ptVbo, int numParticles, float t, float dT)
{
	// what thread am I ?
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	seed = i;

	if(i >= numParticles)
		return;

	// vbo layout (_12_ float)
	// | 3 position | 3 normal | 4 float color | 2 ST  | 
	//
	// physic usage
	// | position   | impulse  | color         | age,- | 
	//
	
	// get current particles state
	ParticleState currentState;
	currentState.X.x = ptVbo[12*i + 0];
	currentState.X.y = ptVbo[12*i + 1];
	currentState.X.z = ptVbo[12*i + 2];
	currentState.P.x = ptVbo[12*i + 3];
	currentState.P.y = ptVbo[12*i + 4];
	currentState.P.z = ptVbo[12*i + 5];
	currentState.age = ptVbo[12*i + 10];
	
	// check particles lifecycle
	if(currentState.age <= 0.0f || currentState.X.y <= 0.0f)
	{
		// start position + noise
		float3 pos = make_float3(-0.03f, 0.3f, 0.0f) + make_float3(0.02f,0.01f,0.02f) * normalize(noise3D());
	
		// start impulse + noise
		float3 imp = 0.06f * noise() * distractDirection3D(make_float3(0.0f,1.0f,0.0f), 0.1f);
		
		// create a new particle
		ptVbo[12*i + 0] = pos.x;
		ptVbo[12*i + 1] = pos.y;
		ptVbo[12*i + 2] = pos.z;
		ptVbo[12*i + 3] = imp.x;
		ptVbo[12*i + 4] = imp.y;
		ptVbo[12*i + 5] = imp.z;
		ptVbo[12*i + 6] = 1.0f;
		ptVbo[12*i + 7] = 0.8f;
		ptVbo[12*i + 8] = 0.8f;
		ptVbo[12*i + 9] = 0.1f;
		ptVbo[12*i + 10] = 50.0f*noise();
	}
	else 
	{
		// integrate state
		//ParticleState newState = rungeKutta4XP(currentState, make_float3(0,-0.1,0), 1, dT);
		ParticleState newState = euler(currentState, make_float3(0,-0.001,0), 1, dT);
		newState.age = currentState.age - dT;

		// distract position
		//newState.X.x += (2.0f*noise()-1.0f)*0.002*sin(10.0f*newState.X.y);
		//newState.X.z += (2.0f*noise()-1.0f)*0.002*sin(10.0f*newState.X.y);

		// create color according the state
		float4 color = colorForAge(newState.age);

		// write back result
		ptVbo[12*i + 0] = newState.X.x;
		ptVbo[12*i + 1] = newState.X.y;
		ptVbo[12*i + 2] = newState.X.z;
		ptVbo[12*i + 3] = newState.P.x;
		ptVbo[12*i + 4] = newState.P.y;
		ptVbo[12*i + 5] = newState.P.z;
		ptVbo[12*i + 6] = color.x;
		ptVbo[12*i + 7] = color.y;
		ptVbo[12*i + 8] = color.z;
		ptVbo[12*i + 9] = color.w;
		ptVbo[12*i + 10] = newState.age;
	}
}

// host sided interface code
extern "C" 
{
	void launchParticleKernel(float *ptVbo, int numParticles, float t, float dT)
	{
		particleKernel<<<ceil(numParticles/(float)32), 32>>>(ptVbo, numParticles, t, dT);
	}
}