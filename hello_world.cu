#include <cstdlib>
#include "helper_cuda.h"

__global__ void kernel(float* a, float* b, float* c) {
    *c = *a + *b;
}

extern "C" {
    void launchKernel(float* a, float* b, float* c) {
        kernel<<<1,1>>>(a, b, c);
    }
}
