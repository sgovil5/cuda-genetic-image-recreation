#ifndef FITNESS_CUH
#define FITNESS_CUH

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "types.cuh"

// Device functions
__device__ Color blendColors(const Color& color1, const Color& color2);
__host__ __device__ float calculateDeltaE(const Color& color1, const Color& color2);
__device__ bool isInsidePolygon(float x, float y, const Polygon* polygon);

// Kernel function
__global__ void fitness_kernel(const Image* population, const unsigned char* target_image, float* fitness_scores, int width, int height);

// Host function
thrust::host_vector<float> calculate_fitness(thrust::host_vector<Image> population, const unsigned char* target_image, int width, int height);

#endif // FITNESS_CUH