#include "main.cuh"
#include "random_utils.cuh"

__global__ void generate_image_kernel(Image* population, int width, int height, Color avg_color, curandState* states){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    curandState local_state = states[idx];

    Image img;

    // Set background
    float background_prob = device_uniform_dis(&local_state);
    if (background_prob < 0.4f) img.background = avg_color;
    else if (background_prob < 0.6f) img.background = {0, 0, 0, 255};
    else if (background_prob < 0.8f) img.background = {255, 255, 255, 255};
    else {
        img.background = {device_color_dis(&local_state), device_color_dis(&local_state), device_color_dis(&local_state), 255};
    }

    // Generate polygons
    int num_polygons = device_num_polygons_dis(&local_state);
    img.num_polygons = num_polygons;
    img.polygons = (Polygon*)malloc(num_polygons*sizeof(Polygon));

    if (img.polygons == nullptr) {
        printf("Error: Could not allocate memory for polygons.\n");
        return;
    }

    for(int i=0; i<num_polygons; i++){
        Polygon& polygon = img.polygons[i];
        polygon.color = {device_color_dis(&local_state), device_color_dis(&local_state), device_color_dis(&local_state), device_color_dis(&local_state)};

        int num_points = device_num_points_dis(&local_state);
        polygon.num_points = num_points;

        polygon.points = (Point*)malloc(num_points*sizeof(Point));
        polygon.lines = (Line*)malloc(num_points*sizeof(Line));

        if(polygon.points == nullptr || polygon.lines == nullptr){
            printf("Error: Could not allocate memory for polygon points or lines.\n");
            return;
        }

        for(int j=0; j<num_points; j++){
            polygon.points[j] = {curand(&local_state)%width, curand(&local_state)%height};
        }

        for(int j=0; j<num_points; j++){
            polygon.lines[j] = {polygon.points[j], polygon.points[(j+1)%num_points]};
        }
    }
    population[idx] = img;
}

// TODO: CHECK ORDER OF FREEING
__global__ void free_image_memory(Image* population, int population_size){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<population_size){
        for(int i=0; i<population[idx].num_polygons; i++){
            free(population[idx].polygons[i].points);
            free(population[idx].polygons[i].lines);
        }
        free(population[idx].polygons);
    }
}

__global__ void avg_color_kernel(const uchar3* image, int width, int height, Color* result){
    extern __shared__ Color shared_sum[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;

    Color localColor = {0, 0, 0, 0};

    if(idx < width && idy < height){
        uchar3 pixel = image[idy*width + idx];
        localColor = {pixel.x, pixel.y, pixel.z, 255};
    }

    shared_sum[threadId] = localColor;
    __syncthreads();

    // Reduction pattern
    for(int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1){
        if(threadId < s){
            shared_sum[threadId].r += shared_sum[threadId + s].r;
            shared_sum[threadId].g += shared_sum[threadId + s].g;
            shared_sum[threadId].b += shared_sum[threadId + s].b;
        }
        __syncthreads();
    }

    if(threadId == 0){
        atomicAdd(&result->r, shared_sum[0].r);
        atomicAdd(&result->g, shared_sum[0].g);
        atomicAdd(&result->b, shared_sum[0].b);
        result->a = 255;
    }
}

Color calculateAvgColor(const unsigned char* image_data, int width, int height, int channels){
    // Device memory
    uchar3* d_image;
    Color* d_result;
    cudaMalloc(&d_image, width*height*sizeof(uchar3));
    cudaMalloc(&d_result, sizeof(Color));
    
    // Put image in device
    cudaMemcpy(d_image, image_data, width*height*sizeof(uchar3), cudaMemcpyHostToDevice);
    
    // Initialize result in device
    Color h_result = {0, 0, 0, 0};
    cudaMemcpy(d_result, &h_result, sizeof(Color), cudaMemcpyHostToDevice);

    //Setup grids and blocks
    dim3 block(16, 16);
    dim3 grid((width + block.x-1)/block.x, (height + block.y-1)/block.y);

    int sharedMemSize = block.x*block.y*sizeof(Color);

    avg_color_kernel<<<grid, block, sharedMemSize>>>(d_image, width, height, d_result);

    // Check for kernel launch errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        throw std::runtime_error("Kernel launch failed");
    }

    cudaMemcpy(&h_result, d_result, sizeof(Color), cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_result);

    int total_pixels = width * height;
    h_result.r /= total_pixels;
    h_result.g /= total_pixels;
    h_result.b /= total_pixels;

    return h_result;
}

thrust::host_vector<Image> init_population(int population_size, int width, int height, Color avg_color){
    Image* d_population;
    curandState* d_states;

    cudaMalloc(&d_population, population_size*sizeof(Image));
    cudaMalloc(&d_states, population_size*sizeof(curandState));

    int block_size = 256;
    int grid_size = (population_size+block_size-1)/block_size;

    init_curand_states<<<grid_size, block_size>>>(d_states, unsigned(time(NULL)));

    generate_image_kernel<<<grid_size, block_size>>>(d_population, width, height, avg_color, d_states);

    // Check for kernel errors
    cudaError_t cudaStatus = cudaGetLastError();
    if(cudaStatus != cudaSuccess){
        std::cerr<<"Kernel launched failed: "<<cudaGetErrorString(cudaStatus)<<std::endl;
        throw std::runtime_error("Kernel launch failed");
    }

    // copy results back to host (thrust is fast)
    thrust::device_vector<Image> d_vec(d_population, d_population + population_size);
    thrust::host_vector<Image> h_vec = d_vec;

    free_image_memory<<<grid_size, block_size>>>(d_population, population_size);

    cudaFree(d_population);
    cudaFree(d_states);

    return h_vec;
}

int main() {
    int width, height, channels;
    unsigned char* image_data = stbi_load("image.png", &width, &height, &channels, 3);
    if(!image_data){
        std::cerr<<"Error in loading the image"<<std::endl;
        exit(1);
    }

    Color avg_color = calculateAvgColor(image_data, width, height, channels);

    int population_size = 5000;
    thrust::host_vector<Image> population = init_population(population_size, width, height, avg_color);
    
    
    std::cout<<"complete"<<std::endl;



    return 0;
}