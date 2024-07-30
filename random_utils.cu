#include "random_utils.cuh"

__global__ void init_curand_states(curandState* states, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}
