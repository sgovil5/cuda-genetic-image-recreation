#ifndef FITNESS_CUH
#define FITNESS_CUH

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include "types.cuh"
#include "ColorUtils.cuh"

__device__ ColorUtils::xyzColor rgbToXyz(ColorUtils::rgbColor c);
__device__ ColorUtils::CIELABColorSpace xyzToLab(ColorUtils::xyzColor c);
__device__ float getColorDeltaE(ColorUtils::rgbColor c1, ColorUtils::rgbColor c2);

__global__ void calculate_fitness_kernel(Image* population, Image* original_image, float* fitness_scores);

thrust::host_vector<float> calculate_fitness(thrust::host_vector<Image> population, Image original_image);

#endif // FITNESS_CUH