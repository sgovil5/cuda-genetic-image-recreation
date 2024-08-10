#include "initialize.cuh"

Color calculate_avg_color(Image image){
    long long total_r = 0, total_g = 0, total_b = 0;
    int total_pixels = image.width * image.height;

    for (int i = 0; i < total_pixels * image.channels; i += image.channels) {
        total_r += image.data[i];
        total_g += image.data[i + 1];
        total_b += image.data[i + 2];
    }

    return Color{
        static_cast<float>(total_r / total_pixels),
        static_cast<float>(total_g / total_pixels),
        static_cast<float>(total_b / total_pixels),
        255
    };
}

__device__ Polygon create_polygon(curandState* state) {
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

__global__ void generate_image_kernel(InitialImage* population, Color avg_color, curandState* states, unsigned char* image_buffers){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= POPULATION_SIZE) return;
    curandState local_state = states[idx];

    InitialImage img;

    float background_prob = device_uniform_dis(&local_state);
    if (background_prob < 0.4f) img.background = avg_color;
    else if (background_prob < 0.6f) img.background = {0, 0, 0, 255};
    else if (background_prob < 0.8f) img.background = {255, 255, 255, 255};
    else{
        img.background = {
            device_uniform_dis(&local_state)*255,
            device_uniform_dis(&local_state)*255,
            device_uniform_dis(&local_state)*255,
            255
        };
    }

    img.num_polygons = curand_uniform(&local_state) * 3 + 3; // 3-6 polygons
    
    for(int i=0; i<img.num_polygons; i++){
        img.polygons[i] = create_polygon(&local_state);
    }

    population[idx] = img;
}

thrust::host_vector<InitialImage> init_population(Image original_image){
    // Calculate average image color
    Color avg_color = calculate_avg_color(original_image);

    InitialImage* d_population;
    curandState* d_states;
    unsigned char* d_image_buffers;
    

    cudaMalloc(&d_population, POPULATION_SIZE*sizeof(InitialImage));
    cudaMalloc(&d_states, POPULATION_SIZE*sizeof(curandState));
    cudaMalloc(&d_image_buffers, POPULATION_SIZE*WIDTH*HEIGHT*3*sizeof(unsigned char));

    CUDA_CHECK(cudaMalloc(&d_population, POPULATION_SIZE*sizeof(InitialImage)));
    CUDA_CHECK(cudaMalloc(&d_states, POPULATION_SIZE*sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_image_buffers, POPULATION_SIZE*WIDTH*HEIGHT*3*sizeof(unsigned char)));

    int block_size = 256;
    int grid_size = (POPULATION_SIZE + block_size - 1) / block_size;

    init_curand_states<<<grid_size, block_size>>>(d_states, unsigned(time(NULL)), POPULATION_SIZE);

    generate_image_kernel<<<grid_size, block_size>>>(d_population, avg_color, d_states, d_image_buffers);

    thrust::host_vector<InitialImage> population(POPULATION_SIZE);
    cudaMemcpy(population.data(), d_population, POPULATION_SIZE*sizeof(InitialImage), cudaMemcpyDeviceToHost);

    cudaFree(d_population);
    cudaFree(d_states);
    cudaFree(d_image_buffers);

    return population;
}