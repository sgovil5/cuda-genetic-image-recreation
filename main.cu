#include "main.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"



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
        // stbi_write_png(("output_image_" + std::to_string(i) + ".png").c_str(), WIDTH, HEIGHT, channels, image_buffer, WIDTH * channels);
    }

    thrust::host_vector<float> fitness_scores = calculate_fitness(image_buffers, original_image);

    thrust::host_vector<Image> new_population = tournament_selection(image_buffers, fitness_scores);

    introduce_mutation(new_population);

    for(int i = 0; i < min(POPULATION_SIZE, 10); i++) {
        stbi_write_png(("new_output_image_" + std::to_string(i) + ".png").c_str(), 
                       WIDTH, HEIGHT, channels, new_population[i].data, WIDTH * channels);
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