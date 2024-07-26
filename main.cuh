#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "types.cuh"

//Kernel functions
__global__ void init_curand_states(curandState* states, unsigned long long seed);
__global__ void generate_image_kernel(Image* population, int width, int height, Color avg_color);

__host__ thrust::host_vector<Image> init_population(int population_size, int width, int height, Color avg_color);
