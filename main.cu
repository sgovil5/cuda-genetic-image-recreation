#include "visualize.cuh"
#include "initialize.cuh"
#include "random_utils.cuh"
#include "fitness.cuh"
#include "main.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

int main() {
    int width, height, channels;
    unsigned char* image_data = stbi_load("image.png", &width, &height, &channels, 3);
    if(!image_data){
        std::cerr<<"Error in loading the image"<<std::endl;
        exit(1);
    }

    Color avg_color = calculateAvgColor(image_data, width, height, channels);
    
    thrust::host_vector<Image> population = init_population(POPULATION_SIZE, width, height, avg_color);
    thrust::host_vector<float> fitness_scores = calculate_fitness(population, image_data, width, height);

    for(int i=0; i<POPULATION_SIZE; ++i){
        visualize_image(population[i], width, height, std::string("test" + std::to_string(i) + ".png").c_str());
        std::cout<<"Fitness score for image "<<i<<" is "<<fitness_scores[i]<<std::endl;
    }
    return 0;
}