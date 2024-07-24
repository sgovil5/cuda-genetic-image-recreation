#ifndef RANDOM_UTILS_CUH
#define RANDOM_UTILS_CUH

#include <curand_kernel.h>

__device__ inline unsigned long long device_color_dis(curandState* state) {
    return curand(state) % 256;
}

__device__ inline int device_num_polygons_dis(curandState* state) {
    return curand(state) % 5 + 3;
}

__device__ inline int device_num_points_dis(curandState* state) {
    return curand(state) % 4 + 3;
}

__device__ inline float device_uniform_dis(curandState* state) {
    return curand_uniform(state);
}

__global__ void init_curand_states(curandState* states, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

#endif // RANDOM_UTILS_CUH