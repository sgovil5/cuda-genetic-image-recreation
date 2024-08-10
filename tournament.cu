#include "tournament.cuh"
__global__ void combined_crossover_kernel(Image* population, int* selected_indices, Image* new_population, curandState* states) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int image_idx = blockIdx.z;

    if (x >= WIDTH || y >= HEIGHT || image_idx >= POPULATION_SIZE) return;

    int parent1_idx = selected_indices[image_idx];
    int parent2_idx = selected_indices[POPULATION_SIZE + image_idx];

    if (parent1_idx < 0 || parent1_idx >= POPULATION_SIZE || parent2_idx < 0 || parent2_idx >= POPULATION_SIZE) return;

    curandState local_state = states[image_idx];
    bool do_vertical = curand(&local_state) % 2 == 0;

    int split_point = do_vertical ? WIDTH / 2 : HEIGHT / 2;

    int pixel_idx = (y * WIDTH + x) * 3;
    if (pixel_idx + 2 >= WIDTH * HEIGHT * 3) return; 

    Image* src_image;
    if (do_vertical) {
        src_image = (x < split_point) ? &population[parent1_idx] : &population[parent2_idx];
    } else {
        src_image = (y < split_point) ? &population[parent1_idx] : &population[parent2_idx];
    }
    
    new_population[image_idx].data[pixel_idx] = src_image->data[pixel_idx];
    new_population[image_idx].data[pixel_idx + 1] = src_image->data[pixel_idx + 1];
    new_population[image_idx].data[pixel_idx + 2] = src_image->data[pixel_idx + 2];
}

__global__ void tournament_selection_kernel(Image* population, float* fitness_scores, int* selected_indices, curandState* states){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=2*POPULATION_SIZE) return;

    curandState local_state = states[idx];
    int best_idx = -1;
    float best_fitness = -INFINITY;

    for(int i=0; i<TOURNAMENT_SIZE; i++){
        int candidate = curand(&local_state) % POPULATION_SIZE;

        // hack to avoid selecting 0'th image because for some reason there's memory corruption
        while(candidate==0){
            candidate = curand(&local_state) % POPULATION_SIZE;
        }
        if(fitness_scores[candidate] > best_fitness){
            best_fitness = fitness_scores[candidate];
            best_idx = candidate;
        }
    }

    selected_indices[idx] = best_idx;
}

thrust::host_vector<Image> tournament_selection(thrust::host_vector<Image> population, thrust::host_vector<float> fitness_scores) {
    Image* d_population;
    float* d_fitness_scores;
    int* d_selected_indices;
    Image* d_new_population;
    curandState* d_states;

    cudaMalloc(&d_population, sizeof(Image) * POPULATION_SIZE);
    cudaMalloc(&d_fitness_scores, sizeof(float) * POPULATION_SIZE);
    cudaMalloc(&d_selected_indices, sizeof(int) * 2 * POPULATION_SIZE);
    cudaMalloc(&d_new_population, sizeof(Image) * POPULATION_SIZE);
    cudaMalloc(&d_states, sizeof(curandState) * 2 * POPULATION_SIZE);

    // ALlocate and copy memory for image data
    for(int i=0; i<POPULATION_SIZE; i++){
        unsigned char* d_image_data;
        cudaMalloc(&d_image_data, WIDTH * HEIGHT * 3 * sizeof(unsigned char));
        cudaMemcpy(d_image_data, population[i].data, WIDTH * HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
        Image h_image = {d_image_data, WIDTH, HEIGHT, 3};
        cudaMemcpy(&d_population[i], &h_image, sizeof(Image), cudaMemcpyHostToDevice);
    }

    // Copy fitness scores
    cudaMemcpy(d_fitness_scores, fitness_scores.data(), sizeof(float) * POPULATION_SIZE, cudaMemcpyHostToDevice);

    // Allocate memory for new population (after tournament selection)
    for(int i=0; i<POPULATION_SIZE; i++){
        Image h_new_image;
        h_new_image.width = WIDTH; 
        h_new_image.height = HEIGHT;
        h_new_image.channels = 3;
        cudaMalloc(&h_new_image.data, WIDTH * HEIGHT * 3 * sizeof(unsigned char));
        cudaMemset(h_new_image.data, 0, WIDTH * HEIGHT * 3 * sizeof(unsigned char));
        cudaMemcpy(&d_new_population[i], &h_new_image, sizeof(Image), cudaMemcpyHostToDevice);
    }


    // Init curand states (have to do it again because we're computing 2*POPULATION_SIZE)
    int block_size = 256;
    int grid_size = (2 * POPULATION_SIZE + block_size - 1) / block_size;
    init_curand_states<<<grid_size, block_size>>>(d_states, unsigned(time(NULL)));

    cudaDeviceSynchronize();

    // Select parent indices
    grid_size = (POPULATION_SIZE + block_size - 1) / block_size;
    tournament_selection_kernel<<<grid_size, block_size>>>(d_population, d_fitness_scores, d_selected_indices, d_states);

    cudaDeviceSynchronize();
    
    // Vertical crossover
    dim3 crossover_block_size(16, 16, 1);
    dim3 crossover_grid_size(
        (WIDTH + crossover_block_size.x - 1) / crossover_block_size.x,
        (HEIGHT + crossover_block_size.y - 1) / crossover_block_size.y,
        POPULATION_SIZE
    );

    combined_crossover_kernel<<<crossover_grid_size, crossover_block_size>>>(d_population, d_selected_indices, d_new_population, d_states);
    cudaDeviceSynchronize();

    // Copy new population to host
    thrust::host_vector<Image> new_population(POPULATION_SIZE);
    for(int i=0; i<POPULATION_SIZE; i++){
        Image h_image;
        CUDA_CHECK(cudaMemcpy(&h_image, &d_new_population[i], sizeof(Image), cudaMemcpyDeviceToHost));
        
        unsigned char* h_image_data = new unsigned char[WIDTH * HEIGHT * 3];
        CUDA_CHECK(cudaMemcpy(h_image_data, h_image.data, WIDTH * HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        new_population[i] = {h_image_data, WIDTH, HEIGHT, 3};
    }

    // Free device memory
    for (int i = 0; i < POPULATION_SIZE; i++) {
        Image h_image;
        CUDA_CHECK(cudaMemcpy(&h_image, &d_population[i], sizeof(Image), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(h_image.data));
        
        CUDA_CHECK(cudaMemcpy(&h_image, &d_new_population[i], sizeof(Image), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(h_image.data));
    }

    CUDA_CHECK(cudaFree(d_population));
    CUDA_CHECK(cudaFree(d_fitness_scores));
    CUDA_CHECK(cudaFree(d_selected_indices));
    CUDA_CHECK(cudaFree(d_new_population));
    CUDA_CHECK(cudaFree(d_states));

    return new_population;
}