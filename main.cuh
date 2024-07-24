#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

struct Color {
    unsigned long long r, g, b;
    unsigned long long a;
};

struct Point {
    unsigned int x, y;
};

struct Line{
    Point p1, p2;
};

struct Polygon {
    Color color;
    Point* points;
    Line* lines;
    int num_points;
};

struct Image {
    Color background;
    Polygon* polygons;
    int num_polygons;
};

//Kernel functions
__global__ void init_curand_states(curandState* states, unsigned long long seed);
__global__ void generate_image_kernel(Image* population, int width, int height, Color avg_color);

__host__ thrust::host_vector<Image> init_population(int population_size, int width, int height, Color avg_color);
