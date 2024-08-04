#include "visualize.cuh"

// Atomic add for colors (make sure value doesn't go over 255)
__device__ void atomicAddRGB(unsigned char* address, unsigned char val){
    int* intAddress = (int*)((void*) address);
    int old = *intAddress;
    int assumed;
    do{
        assumed = old;
        int newVal = min(255, ((assumed & 0xff) + val));
        old = atomicCAS(intAddress, assumed, (assumed & 0xffffff00) | newVal); 
    } while(assumed != old);
}

__device__ void device_find_intersections(const Polygon& polygon, int y, int* nodes, int* nodeCnt) {
    int j = polygon.num_points - 1;
    for (int i = 0; i < polygon.num_points; i++) {
        if ((polygon.points[i].y < y && polygon.points[j].y >= y) || 
            (polygon.points[j].y < y && polygon.points[i].y >= y)) {
            int idx = atomicAdd(nodeCnt, 1);
            nodes[idx] = polygon.points[i].x + (y - polygon.points[i].y) * (polygon.points[j].x - polygon.points[i].x) / 
                (polygon.points[j].y - polygon.points[i].y + 0.0001f);
        }
        j = i;
    }
}

__device__ void device_fill_scanline(unsigned char* buffer, int width, int height, int y, int x1, int x2, const Color& color) {
    if (x1 > x2) { int temp = x1; x1 = x2; x2 = temp; }
    for (int x = x1; x <= x2; ++x) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            int index = (y * width + x) * 3;
            atomicAddRGB(&buffer[index],(unsigned char) color.r);
            atomicAddRGB(&buffer[index + 1],(unsigned char) color.g);
            atomicAddRGB(&buffer[index + 2],(unsigned char) color.b);
        }
    }
}

__device__ void device_draw_line(unsigned char* buffer, int width, int height, const Line& line, const Color& color) {
    int x0 = line.p1.x, y0 = line.p1.y;
    int x1 = line.p2.x, y1 = line.p2.y;
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;

    float alpha = color.a / 255.0f;
    float invAlpha = 1.0f - alpha;

    while (true) {
        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
            int index = (y0 * width + x0) * 3;
            
            // Blend the line color with the existing color in the buffer
            buffer[index] = (unsigned char)(alpha * color.r + invAlpha * buffer[index]);
            buffer[index + 1] = (unsigned char)(alpha * color.g + invAlpha * buffer[index + 1]);
            buffer[index + 2] = (unsigned char)(alpha * color.b + invAlpha * buffer[index + 2]);
        }
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

__device__ bool isInside(float x, float y, const Polygon* polygon) {
    int i, j;
    bool inside = false;
    for (i = 0, j = polygon->num_points - 1; i < polygon->num_points; j = i++) {
        float xi = polygon->points[i].x, yi = polygon->points[i].y;
        float xj = polygon->points[j].x, yj = polygon->points[j].y;
        
        if (((yi > y) != (yj > y)) &&
            (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }
    return inside;
}

__device__ void blendColor(unsigned char* dst, const Color& src) {
    float alpha = src.a / 255.0f;
    float invAlpha = 1.0f - alpha;
    
    dst[0] = (unsigned char)(alpha * src.r + invAlpha * dst[0]);
    dst[1] = (unsigned char)(alpha * src.g + invAlpha * dst[1]);
    dst[2] = (unsigned char)(alpha * src.b + invAlpha * dst[2]);
}

__global__ void fill_polygon_parallel(unsigned char* buffer, int width, int height, const Polygon* polygon) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    if (isInside(x + 0.5f, y + 0.5f, polygon)) {
        int index = (y * width + x) * 3;
        blendColor(&buffer[index], polygon->color);
    }
}

__global__ void draw_polygon_edges(unsigned char* buffer, int width, int height, const Polygon* polygon) {
    int i = threadIdx.x;
    if (i < polygon->num_points) {
        device_draw_line(buffer, width, height, polygon->lines[i], polygon->color);
    }
}

unsigned char* visualize_image(InitialImage& img, int width, int height) {
    unsigned char* h_buffer = new unsigned char[width * height * 3];
    unsigned char* d_buffer;
    cudaMalloc(&d_buffer, width * height * 3 * sizeof(unsigned char));

    // Fill background
    for (int i = 0; i < width * height; ++i) {
        h_buffer[i * 3] = img.background.r;
        h_buffer[i * 3 + 1] = img.background.g;
        h_buffer[i * 3 + 2] = img.background.b;
    }

    cudaMemcpy(d_buffer, h_buffer, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    Polygon* d_polygons;
    cudaMalloc(&d_polygons, img.num_polygons*sizeof(Polygon));
    cudaMemcpy(d_polygons, img.polygons, img.num_polygons*sizeof(Polygon), cudaMemcpyHostToDevice);

    // Create a vector of polygon indices and sort based on alpha values (for order of overlaying polygons)
    std::vector<int> polygonOrder(img.num_polygons);
    for (int i = 0; i < img.num_polygons; ++i) {
        polygonOrder[i] = i;
    }

     // Sort polygons based on alpha values (descending order)
    std::sort(polygonOrder.begin(), polygonOrder.end(), 
        [&img](int a, int b) {
            return img.polygons[a].color.a > img.polygons[b].color.a;
        });


    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y);

    // Fill polygons in sorted order
    for (int index : polygonOrder) {
        fill_polygon_parallel<<<gridDim, blockDim>>>(d_buffer, width, height, &d_polygons[index]);
        cudaDeviceSynchronize();
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            // Handle error (e.g., break or return)
        }
    }

    // Draw edges
    for (int p = 0; p < img.num_polygons; ++p) {
        draw_polygon_edges<<<1, MAX_POINTS>>>(d_buffer, width, height, &d_polygons[p]);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_buffer, d_buffer, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_polygons);
    cudaFree(d_buffer);

    return h_buffer;
}
