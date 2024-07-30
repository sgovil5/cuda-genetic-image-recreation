#ifndef RANDOM_UTILS_CUH
#define RANDOM_UTILS_CUH

#include <curand_kernel.h>

__device__ inline float device_color_dis(curandState* state) {
    return floorf(curand_uniform(state) * 256.0f);
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

__global__ void init_curand_states(curandState* states, unsigned long long seed);

#endif // RANDOM_UTILS_CUH