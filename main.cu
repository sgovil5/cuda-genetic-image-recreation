#include "visualize.cuh"
#include "initialize.cuh"
#include "random_utils.cuh"
#include "fitness.cuh"
#include "main.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

__device__ void blendParents(const CrossoverImage* parent1, const CrossoverImage* parent2, Image* child, curandState* state) {
    float x = device_uniform_dis(state);
    
    // Blend background colors
    child->background.r = parent1->background.r * (1 - x) + parent2->background.r * x;
    child->background.g = parent1->background.g * (1 - x) + parent2->background.g * x;
    child->background.b = parent1->background.b * (1 - x) + parent2->background.b * x;
    child->background.a = 255.0f;

    // Combine polygons from both parents
    int total_polygons = parent1->num_polygons + parent2->num_polygons;
    child->num_polygons = min(total_polygons, MAX_POLYGONS);

    int i = 0;
    // Copy polygons from parent1 with adjusted opacity
    for (; i < parent1->num_polygons && i < MAX_POLYGONS; ++i) {
        child->polygons[i] = parent1->polygons[i];
        child->polygons[i].color.a *= (1 - x);
    }
    // Copy polygons from parent2 with adjusted opacity
    for (int j = 0; i < child->num_polygons; ++i, ++j) {
        child->polygons[i] = parent2->polygons[j];
        child->polygons[i].color.a *= x;
    }

    // Initialize fitness score to 0
    child->fitness_score = 0;
}

__global__ void tournament_select_kernel(Image* population, Image* offspring, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= POPULATION_SIZE) return;

    // Initialize curand state for this thread
    curand_init(clock64(), idx, 0, &states[idx]);
    
    // Select first parent
    CrossoverImage parent1;
    int best_fitness1 = -1;
    for (int i = 0; i < TOURNAMENT_SIZE; ++i) {
        int random_idx = (int)(device_uniform_dis(&states[idx]) * POPULATION_SIZE);
        if (best_fitness1 == -1 || population[random_idx].fitness_score > best_fitness1) {
            parent1.background = population[random_idx].background;
            parent1.num_polygons = population[random_idx].num_polygons;
            parent1.polygons = population[random_idx].polygons;
            best_fitness1 = population[random_idx].fitness_score;
        }
    }

    // Select second parent
    CrossoverImage parent2;
    int best_fitness2 = -1;
    for (int i = 0; i < TOURNAMENT_SIZE; ++i) {
        int random_idx = (int)(device_uniform_dis(&states[idx]) * POPULATION_SIZE);
        if (best_fitness2 == -1 || population[random_idx].fitness_score > best_fitness2) {
            parent2.background = population[random_idx].background;
            parent2.num_polygons = population[random_idx].num_polygons;
            parent2.polygons = population[random_idx].polygons;
            best_fitness2 = population[random_idx].fitness_score;
        }
    }
    
    // Blend parents to create child
    blendParents(&parent1, &parent2, &offspring[idx], &states[idx]);
}

thrust::host_vector<Image> tournament_select(thrust::host_vector<Image> population){
    Image* d_population;
    Image* d_offspring;
    curandState* d_states;

    cudaMalloc(&d_population, POPULATION_SIZE*sizeof(Image));
    cudaMalloc(&d_offspring, POPULATION_SIZE*sizeof(Image));
    cudaMalloc(&d_states, POPULATION_SIZE*sizeof(curandState));

    cudaMemcpy(d_population, population.data(), POPULATION_SIZE*sizeof(Image), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (POPULATION_SIZE + blockSize - 1) / blockSize;
    tournament_select_kernel<<<gridSize, blockSize>>>(d_population,d_offspring, d_states);
    // Add error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    cudaMemcpy(population.data(), d_offspring, POPULATION_SIZE*sizeof(Image), cudaMemcpyDeviceToHost);

    cudaFree(d_population);
    cudaFree(d_offspring);
    cudaFree(d_states);

    return population;
}

int main() {
    int width, height, channels;
    unsigned char* image_data = stbi_load("image.png", &width, &height, &channels, 3);
    if(!image_data){
        std::cerr<<"Error in loading the image"<<std::endl;
        exit(1);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;
    // Timing for calculateAvgColor
    cudaEventRecord(start);
    Color avg_color = calculateAvgColor(image_data, width, height, channels);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "calculateAvgColor time: " << milliseconds << " ms" << std::endl;

    // Timing for init_population
    cudaEventRecord(start);
    thrust::host_vector<Image> population = init_population(POPULATION_SIZE, width, height, avg_color);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "init_population time: " << milliseconds << " ms" << std::endl;

    // Timing for calculate_fitness
    cudaEventRecord(start);
    thrust::host_vector<float> fitness_scores = calculate_fitness(population, image_data, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "calculate_fitness time: " << milliseconds << " ms" << std::endl;

    // Timing for tournament_select
    cudaEventRecord(start);
    population = tournament_select(population);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "tournament_select time: " << milliseconds << " ms" << std::endl;

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}