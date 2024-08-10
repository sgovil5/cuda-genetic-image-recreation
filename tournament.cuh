#ifndef TOURNAMENT_CUH
#define TOURNAMENT_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include "types.cuh"

__global__ void combined_crossover_kernel(Image* population, int* selected_indices, Image* new_population, curandState* states);

__global__ void tournament_selection_kernel(Image* population, float* fitness_scores, int* selected_indices, curandState* states);

__global__ void init_curand_states(curandState* states, unsigned long long seed);

thrust::host_vector<Image> tournament_selection(thrust::host_vector<Image> population, thrust::host_vector<float> fitness_scores);

#endif // TOURNAMENT_CUH