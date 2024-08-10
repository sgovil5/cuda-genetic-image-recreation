#include "mutation.cuh"

__device__ Polygon create_polygon_mutation(curandState* state) {
    Polygon polygon;
    polygon.color = {
        device_uniform_dis(state) * 255,
        device_uniform_dis(state) * 255,
        device_uniform_dis(state) * 255,
        device_uniform_dis(state) * 255
    };

    int num_points = curand_uniform(state) * 3 + 3; // 3-6 points
    polygon.num_points = num_points;

    for (int i = 0; i < num_points; i++) {
        polygon.points[i].x = device_uniform_dis(state) * (WIDTH - 1);
        polygon.points[i].y = device_uniform_dis(state) * (HEIGHT - 1);
    }

    for(int i=0; i<num_points; i++){
        polygon.lines[i].p1 = polygon.points[i];
        polygon.lines[i].p2 = polygon.points[(i+1)%num_points];
    }

    return polygon;
}

__device__ void calculate_bounding_box(const Polygon& polygon, int& minX, int& minY, int& maxX, int& maxY) {
    minX = WIDTH - 1;
    minY = HEIGHT - 1;
    maxX = 0;
    maxY = 0;
    
    for (int i = 0; i < polygon.num_points; i++) {
        minX = min(minX, (int)polygon.points[i].x);
        minY = min(minY, (int)polygon.points[i].y);
        maxX = max(maxX, (int)polygon.points[i].x);
        maxY = max(maxY, (int)polygon.points[i].y);
    }
    
    // Clamp to image boundaries
    minX = max(0, minX);
    minY = max(0, minY);
    maxX = min(WIDTH - 1, maxX);
    maxY = min(HEIGHT - 1, maxY);
}

__device__ bool is_inside_mutation(float x, float y, const Polygon* polygon) {
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

__device__ void blend_color_mutation(unsigned char* dst, const Color& src) {
    float alpha = src.a / 255.0f;
    float invAlpha = 1.0f - alpha;
    
    dst[0] = (unsigned char)(alpha * src.r + invAlpha * dst[0]);
    dst[1] = (unsigned char)(alpha * src.g + invAlpha * dst[1]);
    dst[2] = (unsigned char)(alpha * src.b + invAlpha * dst[2]);
}

__device__ void draw_line_mutation(unsigned char* buffer, int width, int height, const Line& line, const Color& color) {
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


__global__ void mutation_kernel(Image* population, curandState* states){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=POPULATION_SIZE) return;

    curandState local_state = states[idx];

    bool mutate = curand_uniform(&local_state) < MUTATION_RATE;

    if(!mutate) return;

    Polygon generated_polygon = create_polygon_mutation(&local_state);

    // Calculate bounding box
    int minX, minY, maxX, maxY;
    calculate_bounding_box(generated_polygon, minX, minY, maxX, maxY);

    // Blend the generated polygon with the image within the bounding box
    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            if (is_inside_mutation(x + 0.5f, y + 0.5f, &generated_polygon)) {
                int index = (y * WIDTH + x) * 3;
                blend_color_mutation(&population[idx].data[index], generated_polygon.color);
            }
        }
    }

    // Draw the polygon edges
    for (int i = 0; i < generated_polygon.num_points; ++i) {
        draw_line_mutation(population[idx].data, WIDTH, HEIGHT, generated_polygon.lines[i], generated_polygon.color);
    }
}

void introduce_mutation(thrust::host_vector<Image>& population){
    Image* d_population;
    curandState* d_states;

    CUDA_CHECK(cudaMalloc(&d_population, sizeof(Image) * POPULATION_SIZE));
    CUDA_CHECK(cudaMalloc(&d_states, sizeof(curandState) * POPULATION_SIZE));

    // Allocate and copy memory for image data
    for(int i=0; i<POPULATION_SIZE; i++){
        unsigned char* d_image_data;
        CUDA_CHECK(cudaMalloc(&d_image_data, WIDTH * HEIGHT * 3 * sizeof(unsigned char)));
        CUDA_CHECK(cudaMemcpy(d_image_data, population[i].data, WIDTH * HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
        Image h_image = {d_image_data, WIDTH, HEIGHT, 3};
        CUDA_CHECK(cudaMemcpy(&d_population[i], &h_image, sizeof(Image), cudaMemcpyHostToDevice));
    }

    int block_size = 256;
    int grid_size = (POPULATION_SIZE + block_size - 1) / block_size;

    init_curand_states<<<grid_size, block_size>>>(d_states, unsigned(time(NULL)), POPULATION_SIZE);

    CUDA_CHECK(cudaDeviceSynchronize());

    mutation_kernel<<<grid_size, block_size>>>(d_population, d_states);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back the mutated population to host
    for(int i=0; i<POPULATION_SIZE; i++){
        Image h_image;
        CUDA_CHECK(cudaMemcpy(&h_image, &d_population[i], sizeof(Image), cudaMemcpyDeviceToHost));

        unsigned char* h_image_data = new unsigned char[WIDTH * HEIGHT * 3];
        CUDA_CHECK(cudaMemcpy(h_image_data, h_image.data, WIDTH * HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        population[i] = {h_image_data, WIDTH, HEIGHT, 3};

        // Free the device memory for each image
        CUDA_CHECK(cudaFree(h_image.data));
    }

    CUDA_CHECK(cudaFree(d_population));
    CUDA_CHECK(cudaFree(d_states));
}