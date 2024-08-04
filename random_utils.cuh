#ifndef RANDOM_UTILS_CUH
#define RANDOM_UTILS_CUH

#include <curand_kernel.h>

__device__ inline int device_int_dis(curandState* state) {
    return curand(state);
}

__device__ inline float device_uniform_dis(curandState* state) {
    return curand_uniform(state);
}

__global__ void init_curand_states(curandState* states, unsigned long long seed);

#endif // RANDOM_UTILS_CUH