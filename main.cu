#include "main.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

__global__ void vertical_crossover_kernel(Image* population, int* selected_indices, Image* new_population) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int image_idx = blockIdx.z;

    if (x >= WIDTH || y >= HEIGHT || image_idx >= POPULATION_SIZE) return;

    int parent1_idx = selected_indices[image_idx];
    int parent2_idx = selected_indices[POPULATION_SIZE + image_idx];

    if (parent1_idx < 0 || parent1_idx >= POPULATION_SIZE || parent2_idx < 0 || parent2_idx >= POPULATION_SIZE) return;

    Image parent1 = population[parent1_idx];
    Image parent2 = population[parent2_idx];

    int split_point = WIDTH / 2;

    int pixel_idx = (y * WIDTH + x) * 3;
    if (pixel_idx >= WIDTH * HEIGHT * 3) return;

    if (x < split_point) {
        new_population[image_idx].data[pixel_idx] = parent1.data[pixel_idx];
        new_population[image_idx].data[pixel_idx + 1] = parent1.data[pixel_idx + 1];
        new_population[image_idx].data[pixel_idx + 2] = parent1.data[pixel_idx + 2];
    } else {
        new_population[image_idx].data[pixel_idx] = parent2.data[pixel_idx];
        new_population[image_idx].data[pixel_idx + 1] = parent2.data[pixel_idx + 1];
        new_population[image_idx].data[pixel_idx + 2] = parent2.data[pixel_idx + 2];
    }
}

__global__ void tournament_selection_kernel(Image* population, float* fitness_scores, int* selected_indices, curandState* states){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=POPULATION_SIZE) return;

    curandState local_state = states[idx];
    int best_idx = -1;
    float best_fitness = -INFINITY;

    for(int i=0; i<TOURNAMENT_SIZE; i++){
        int candidate = curand(&local_state) % POPULATION_SIZE;
        if(fitness_scores[candidate] > best_fitness){
            best_fitness = fitness_scores[candidate];
            best_idx = candidate;
        }
    }

    selected_indices[idx] = best_idx;
}

thrust::host_vector<Image> tournament_selection(thrust::host_vector<Image>& population, thrust::host_vector<float>& fitness_scores) {
    cudaError_t err;
    
    std::cout << "Tournament selection started. Population size: " << POPULATION_SIZE << std::endl;
    if (population.size() != POPULATION_SIZE) {
        std::cerr << "Error: population size mismatch. Expected " << POPULATION_SIZE << ", got " << population.size() << std::endl;
        std::cerr.flush();
        exit(1);
    }

    if (fitness_scores.size() != POPULATION_SIZE) {
        std::cerr << "Error: fitness_scores size mismatch. Expected " << POPULATION_SIZE << ", got " << fitness_scores.size() << std::endl;
        std::cerr.flush();
        exit(1);
    }

    for (int i = 0; i < POPULATION_SIZE; i++) {
        if (population[i].data == nullptr) {
            std::cerr << "Error: population[" << i << "].data is null" << std::endl;
            std::cerr.flush();
            exit(1);
        }
        if (population[i].width != WIDTH || population[i].height != HEIGHT) {
            std::cerr << "Error: population[" << i << "] has incorrect dimensions. Expected " << WIDTH << "x" << HEIGHT 
                      << ", got " << population[i].width << "x" << population[i].height << std::endl;
            std::cerr.flush();
            exit(1);
        }
    }
    
    std::cout << "Tournament selection started. Population size: " << POPULATION_SIZE << std::endl;
    std::cout.flush();

    Image* d_population;
    float* d_fitness_scores;
    int* d_selected_indices;
    Image* d_new_population;
    curandState* d_states;

    // Allocate device memory
    std::cout << "Allocating device memory..." << std::endl;
    std::cout.flush();

    std::cout << "Allocating d_population..." << std::endl;
    std::cout.flush();
    err = cudaMalloc(&d_population, sizeof(Image) * POPULATION_SIZE);
    if(err != cudaSuccess){
        std::cerr << "Failed to allocate d_population: " << cudaGetErrorString(err) << std::endl;
        std::cerr.flush();
        exit(1);
    }

    std::cout << "Allocating d_fitness_scores..." << std::endl;
    std::cout.flush();
    err = cudaMalloc(&d_fitness_scores, sizeof(float) * POPULATION_SIZE);
    if(err != cudaSuccess){
        std::cerr << "Failed to allocate d_fitness_scores: " << cudaGetErrorString(err) << std::endl;
        std::cerr.flush();
        exit(1);
    }

    std::cout << "Allocating d_selected_indices..." << std::endl;
    std::cout.flush();
    err = cudaMalloc(&d_selected_indices, sizeof(int) * 2 * POPULATION_SIZE);
    if(err != cudaSuccess){
        std::cerr << "Failed to allocate d_selected_indices: " << cudaGetErrorString(err) << std::endl;
        std::cerr.flush();
        exit(1);
    }

    std::cout << "Allocating d_new_population..." << std::endl;
    std::cout.flush();
    err = cudaMalloc(&d_new_population, sizeof(Image) * POPULATION_SIZE);
    if(err != cudaSuccess){
        std::cerr << "Failed to allocate d_new_population: " << cudaGetErrorString(err) << std::endl;
        std::cerr.flush();
        exit(1);
    }

    std::cout << "Allocating d_states..." << std::endl;
    std::cout.flush();
    err = cudaMalloc(&d_states, sizeof(curandState) * 2 * POPULATION_SIZE);
    if(err != cudaSuccess){
        std::cerr << "Failed to allocate d_states: " << cudaGetErrorString(err) << std::endl;
        std::cerr.flush();
        exit(1);
    }

     // Allocate memory for image data in d_population and d_new_population
    for (int i = 0; i < POPULATION_SIZE; i++) {
        std::cout << "Allocating d_population[" << i << "].data..." << std::endl;
        std::cout.flush();
        
        // Temporary pointer to hold the device address
        unsigned char* temp_data;
        err = cudaMalloc(&temp_data, WIDTH * HEIGHT * 3 * sizeof(unsigned char));
        if(err != cudaSuccess){
            std::cerr << "Failed to allocate d_population[" << i << "].data: " << cudaGetErrorString(err) << std::endl;
            std::cerr << "Attempted to allocate " << (WIDTH * HEIGHT * 3 * sizeof(unsigned char)) << " bytes" << std::endl;
            std::cerr.flush();
            exit(1);
        }
        
        // Copy the device pointer to the host struct
        err = cudaMemcpy(&(d_population[i].data), &temp_data, sizeof(unsigned char*), cudaMemcpyHostToDevice);
        if(err != cudaSuccess){
            std::cerr << "Failed to copy d_population[" << i << "].data pointer: " << cudaGetErrorString(err) << std::endl;
            std::cerr.flush();
            exit(1);
        }

        std::cout << "Allocating d_new_population[" << i << "].data..." << std::endl;
        std::cout.flush();
        
        err = cudaMalloc(&temp_data, WIDTH * HEIGHT * 3 * sizeof(unsigned char));
        if(err != cudaSuccess){
            std::cerr << "Failed to allocate d_new_population[" << i << "].data: " << cudaGetErrorString(err) << std::endl;
            std::cerr << "Attempted to allocate " << (WIDTH * HEIGHT * 3 * sizeof(unsigned char)) << " bytes" << std::endl;
            std::cerr.flush();
            exit(1);
        }
        
        // Copy the device pointer to the host struct
        err = cudaMemcpy(&(d_new_population[i].data), &temp_data, sizeof(unsigned char*), cudaMemcpyHostToDevice);
        if(err != cudaSuccess){
            std::cerr << "Failed to copy d_new_population[" << i << "].data pointer: " << cudaGetErrorString(err) << std::endl;
            std::cerr.flush();
            exit(1);
        }
    }

    std::cout << "All device memory allocated successfully." << std::endl;
    std::cout.flush();

    // Copy data to device
    std::cout << "Copying data to device..." << std::endl;
    std::cout.flush();
    CUDA_CHECK(cudaMemcpy(d_population, population.data(), sizeof(Image) * POPULATION_SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fitness_scores, fitness_scores.data(), sizeof(float) * POPULATION_SIZE, cudaMemcpyHostToDevice));

    std::cout<<"d_population and d_fitness_scores copied to device successfully"<<std::endl;
    std::cout.flush();
    // Copy image data using cudaMemcpy2D
    for (int i = 0; i < POPULATION_SIZE; i++) {
        std::cout << "Copying image " << i << " to device..." << std::endl;
        std::cout.flush();

        // Get the device pointer for the current image
        unsigned char* device_data;
        CUDA_CHECK(cudaMemcpy(&device_data, &(d_population[i].data), sizeof(unsigned char*), cudaMemcpyDeviceToHost));

        // Perform the 2D memory copy
        cudaError_t err = cudaMemcpy2D(device_data, WIDTH * 3 * sizeof(unsigned char),
                                       population[i].data, WIDTH * 3 * sizeof(unsigned char),
                                       WIDTH * 3 * sizeof(unsigned char), HEIGHT,
                                       cudaMemcpyHostToDevice);

        if (err != cudaSuccess) {
            std::cerr << "Failed to copy image " << i << " to device: " << cudaGetErrorString(err) << std::endl;
            std::cerr << "Source (host) address: " << (void*)population[i].data << std::endl;
            std::cerr << "Destination (device) address: " << (void*)device_data << std::endl;
            std::cerr << "Width: " << WIDTH << ", Height: " << HEIGHT << ", Channels: 3" << std::endl;
            std::cerr.flush();
            exit(1);
        }

        std::cout << "Image " << i << " copied successfully." << std::endl;
        std::cout.flush();
    }

    std::cout << "All image data copied to device successfully." << std::endl;
    std::cout.flush();

    // Initialize random states
    std::cout << "Initializing random states..." << std::endl;
    std::cout.flush();
    init_curand_states<<<(2 * POPULATION_SIZE + 255) / 256, 256>>>(d_states, 2 * POPULATION_SIZE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform tournament selection
    std::cout << "Performing tournament selection..." << std::endl;
    std::cout.flush();
    tournament_selection_kernel<<<(2 * POPULATION_SIZE + 255) / 256, 256>>>(d_population, d_fitness_scores, d_selected_indices, d_states);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform crossover
    std::cout << "Performing crossover..." << std::endl;
    std::cout.flush();
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                  (HEIGHT + blockSize.y - 1) / blockSize.y,
                  POPULATION_SIZE);
    vertical_crossover_kernel<<<gridSize, blockSize>>>(d_population, d_selected_indices, d_new_population);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Kernels executed. Copying data back to host." << std::endl;
    std::cout.flush();

    // Allocate host vector for new population
    thrust::host_vector<Image> new_population(POPULATION_SIZE);

    // Deep copy the new population to host
    for (int i = 0; i < POPULATION_SIZE; i++) {
        std::cout << "Copying image " << i << " to host..." << std::endl;
        std::cout.flush();

        // Allocate host memory
        unsigned char* host_data = new unsigned char[WIDTH * HEIGHT * 3];
        if (host_data == nullptr) {
            std::cerr << "Failed to allocate host memory for image " << i << std::endl;
            std::cerr.flush();
            exit(1);
        }

        // Get the device pointer for the current image
        unsigned char* device_data;
        CUDA_CHECK(cudaMemcpy(&device_data, &(d_new_population[i].data), sizeof(unsigned char*), cudaMemcpyDeviceToHost));

        // Perform the 2D memory copy
        cudaError_t err = cudaMemcpy2D(host_data, WIDTH * 3 * sizeof(unsigned char),
                                       device_data, WIDTH * 3 * sizeof(unsigned char),
                                       WIDTH * 3 * sizeof(unsigned char), HEIGHT,
                                       cudaMemcpyDeviceToHost);

        if (err != cudaSuccess) {
            std::cerr << "Failed to copy image " << i << " from device to host: " << cudaGetErrorString(err) << std::endl;
            std::cerr << "Source (device) address: " << (void*)device_data << std::endl;
            std::cerr << "Destination (host) address: " << (void*)host_data << std::endl;
            std::cerr << "Width: " << WIDTH << ", Height: " << HEIGHT << ", Channels: 3" << std::endl;
            std::cerr.flush();
            delete[] host_data;  // Clean up allocated memory
            exit(1);
        }

        new_population[i].data = host_data;
        new_population[i].width = WIDTH;
        new_population[i].height = HEIGHT;
        new_population[i].channels = 3;

        std::cout << "Image " << i << " copied successfully to host." << std::endl;
        std::cout.flush();
    }

    std::cout << "All new population data copied to host successfully." << std::endl;
    std::cout.flush();

    std::cout << "Data copied to host. New population size: " << new_population.size() << std::endl;
    std::cout.flush();

    // Free device memory
    std::cout << "Freeing device memory..." << std::endl;
    std::cout.flush();

    // Free image data memory
    for (int i = 0; i < POPULATION_SIZE; i++) {
        unsigned char* device_data;
        
        // Free d_population[i].data
        err = cudaMemcpy(&device_data, &(d_population[i].data), sizeof(unsigned char*), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Error retrieving d_population[" << i << "].data pointer: " << cudaGetErrorString(err) << std::endl;
            std::cerr.flush();
        } else if (device_data != nullptr) {
            err = cudaFree(device_data);
            if (err != cudaSuccess) {
                std::cerr << "Error freeing d_population[" << i << "].data: " << cudaGetErrorString(err) << std::endl;
                std::cerr.flush();
            }
        }
        
        // Free d_new_population[i].data
        err = cudaMemcpy(&device_data, &(d_new_population[i].data), sizeof(unsigned char*), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Error retrieving d_new_population[" << i << "].data pointer: " << cudaGetErrorString(err) << std::endl;
            std::cerr.flush();
        } else if (device_data != nullptr) {
            err = cudaFree(device_data);
            if (err != cudaSuccess) {
                std::cerr << "Error freeing d_new_population[" << i << "].data: " << cudaGetErrorString(err) << std::endl;
                std::cerr.flush();
            }
        }
    }

    // Free other device memory
    if (d_population) {
        err = cudaFree(d_population);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing d_population: " << cudaGetErrorString(err) << std::endl;
            std::cerr.flush();
        }
    }

    if (d_fitness_scores) {
        err = cudaFree(d_fitness_scores);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing d_fitness_scores: " << cudaGetErrorString(err) << std::endl;
            std::cerr.flush();
        }
    }

    if (d_selected_indices) {
        err = cudaFree(d_selected_indices);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing d_selected_indices: " << cudaGetErrorString(err) << std::endl;
            std::cerr.flush();
        }
    }

    if (d_states) {
        err = cudaFree(d_states);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing d_states: " << cudaGetErrorString(err) << std::endl;
            std::cerr.flush();
        }
    }

    if (d_new_population) {
        err = cudaFree(d_new_population);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing d_new_population: " << cudaGetErrorString(err) << std::endl;
            std::cerr.flush();
        }
    }

    std::cout << "Device memory freed." << std::endl;
    std::cout.flush();

    std::cout << "Tournament selection completed." << std::endl;
    std::cout.flush();

    return new_population;
}

int main(){
    cudaSetDevice(0);
    int width=WIDTH, height=HEIGHT, channels;
    
    unsigned char* image_data = stbi_load("image.png", &width, &height, &channels, 3);

    if (!image_data) {
        std::cerr << "Error in loading the image" << std::endl;
        exit(1);
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    Image original_image = {image_data, width, height, channels};

    thrust::host_vector<InitialImage> population = init_population(original_image);

    thrust::host_vector<Image> image_buffers;
    for(int i=0; i<POPULATION_SIZE; i++){
        InitialImage curr_img = population[i];
        unsigned char* image_buffer = visualize_image(curr_img, WIDTH, HEIGHT);
        image_buffers.push_back({image_buffer, WIDTH, HEIGHT, channels});
        stbi_write_png(("output_image_" + std::to_string(i) + ".png").c_str(), WIDTH, HEIGHT, channels, image_buffer, WIDTH * channels);
    }

    thrust::host_vector<float> fitness_scores = calculate_fitness(image_buffers, original_image);

    thrust::host_vector<Image> new_population = tournament_selection(image_buffers, fitness_scores);
    for(int i = 0; i < POPULATION_SIZE; i++) {
        if(new_population[i].data != nullptr) {
            for (int j = 0; j < std::min(10, WIDTH * HEIGHT); j++) {
                unsigned char* pixel = &new_population[i].data[j * channels];
                printf("Image %d, Pixel %d: R=%d, G=%d, B=%d\n", i, j, pixel[0], pixel[1], pixel[2]);
            }
            stbi_write_png(("new_output_image_" + std::to_string(i) + ".png").c_str(), 
                           WIDTH, HEIGHT, channels, new_population[i].data, WIDTH * channels);
        } else {
            std::cerr << "Error: Image data is null for image " << i << std::endl;
        }
    }
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total execution time: %f ms\n", milliseconds);

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}