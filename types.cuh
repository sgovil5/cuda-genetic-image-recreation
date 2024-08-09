#ifndef TYPES_CUH
#define TYPES_CUH

#define POPULATION_SIZE 1000
#define WIDTH 100
#define HEIGHT 100

#define TOURNAMENT_SIZE 80

#define MAX_POINTS 8
#define MAX_POLYGONS 8

struct Image {
    unsigned char* data;
    int width;
    int height;
    int channels;
};

struct Color {
    float r, g, b, a;
};

struct Point {
    unsigned int x, y;
};

struct Line {
    Point p1, p2;
};

struct Polygon {
    Color color;
    Point points[MAX_POINTS];
    Line lines[MAX_POINTS];
    int num_points;
};

struct InitialImage {
    Color background;
    Polygon polygons[MAX_POLYGONS];
    int num_polygons;
};

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#endif // TYPES_CUH