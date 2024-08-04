#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "stb/stb_image_write.h"

#ifndef VISUALIZE_CUH
#define VISUALIZE_CUH

#include "types.cuh"

__device__ void atomicAddRGB(unsigned char* address, unsigned char val);
__device__ void device_find_intersections(const Polygon& polygon, int y, int* nodes, int* nodeCnt);
__device__ void device_fill_scanline(unsigned char* buffer, int width, int height, int y, int x1, int x2, const Color& color);
__device__ void device_draw_line(unsigned char* buffer, int width, int height, const Line& line, const Color& color);
__device__ bool isInside(float x, float y, const Polygon* polygon);
__device__ void blendColor(unsigned char* dst, const Color& src);
__global__ void fill_polygon_parallel(unsigned char* buffer, int width, int height, const Polygon* polygon);
__global__ void draw_polygon_edges(unsigned char* buffer, int width, int height, const Polygon* polygon);
unsigned char* visualize_image(InitialImage& img, int width, int height);

#endif 