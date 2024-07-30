#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifndef INITIALIZE_CUH
#define INITIALIZE_CUH

#include "types.cuh"
#include "random_utils.cuh"

__global__ void generate_image_kernel(Image* population, int width, int height, Color avg_color, curandState* states);
__global__ void avg_color_kernel(const uchar3* image, int width, int height, Color* result);
Color calculateAvgColor(const unsigned char* image_data, int width, int height, int channels);
thrust::host_vector<Image> init_population(int population_size, int width, int height, Color avg_color);

#endif