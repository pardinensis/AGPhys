#include <cuda_runtime.h>

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
    float cosTheta = sqrtf(N.x/2+.5f);
    float sinTheta = sqrtf(1.0f-cosTheta*cosTheta);
    float3 localSampling = make_float3(sinTheta*cosf(phi), cosTheta, sinTheta*sinf(phi));
    localSampling.x *= range;
    localSampling.z *= range;

//	// transform to worldspace
//	float3 y = vIn;
//	float3 h = y;
//	if(fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z))
//	{
//		h.x = 1.0f;
//	}
//	else if(fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z))
//	{
//		h.y = 1.0f;
//	}
//	else
//	{
//		h.z = 1.0f;
//	}

//	float3 x = normalize(cross(h,y));
//	float3 z = normalize(cross(x,y));

//	return normalize(localSampling.x * x + localSampling.y * y + localSampling.z * z);
    return localSampling;
}
