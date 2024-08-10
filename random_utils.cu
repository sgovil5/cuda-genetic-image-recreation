#include "random_utils.cuh"

__global__ void init_curand_states(curandState* states, unsigned long long seed, int total_threads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= total_threads) return;
    curand_init(seed, idx, 0, &states[idx]);
}
