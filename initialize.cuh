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

Color calculate_avg_color(Image image);
__device__ Polygon create_polygon(curandState* state);
__global__ void generate_image_kernel(InitialImage* population, Color avg_color, curandState* states, unsigned char* image_buffers);
thrust::host_vector<InitialImage> init_population(Image original_image);


#endif