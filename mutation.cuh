#ifndef MUTATION_CUH
#define MUTATION_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include "types.cuh"
#include "random_utils.cuh"

__device__ Polygon create_polygon_mutation(curandState* state);

__device__ void calculate_bounding_box(const Polygon& polygon, int& minX, int& minY, int& maxX, int& maxY);

__device__ bool is_inside_mutation(float x, float y, const Polygon* polygon);

__device__ void blend_color_mutation(unsigned char* dst, const Color& src);

__device__ void draw_line_mutation(unsigned char* buffer, int width, int height, const Line& line, const Color& color);

__global__ void mutation_kernel(Image* population, curandState* states);

void introduce_mutation(thrust::host_vector<Image>& population);

#endif // MUTATION_CUH