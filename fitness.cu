#include "fitness.cuh"

__device__ Color blendColors(const Color& background, const Color& foreground) {
    float alpha = foreground.a / 255.0f;
    float invAlpha = 1.0f - alpha;

    Color blended_color;
    blended_color.r = background.r * invAlpha + foreground.r * alpha;
    blended_color.g = background.g * invAlpha + foreground.g * alpha;
    blended_color.b = background.b * invAlpha + foreground.b * alpha;
    blended_color.a = background.a * invAlpha + foreground.a * alpha;

    return blended_color;
}

__host__ __device__ float calculateDeltaE(const Color& color1, const Color& color2) {
    float deltaR = (color1.r - color2.r) / 255.0f;
    float deltaG = (color1.g - color2.g) / 255.0f;
    float deltaB = (color1.b - color2.b) / 255.0f;
    
    float result = sqrtf(deltaR * deltaR + deltaG * deltaG + deltaB * deltaB);
    return result;
}

__device__ bool isInsidePolygon(float x, float y, const Polygon* polygon) {
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

__global__ void fitness_kernel(const Image* population, const unsigned char* target_image, float* fitness_scores, int width, int height){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=POPULATION_SIZE) return;

    const Image& img = population[idx];
    float local_delta_e = 0.0f;
    
    for(int y=0; y<height; ++y){
        for(int x=0; x<width; ++x){
            int imageIndex = (y*width+x)*3;
            Color target_color = {(float) target_image[imageIndex], (float) target_image[imageIndex+1], (float) target_image[imageIndex+2], 255};
            Color pixel_color = img.background;
            bool polygon_applied = false;
            for(int p=0; p<img.num_polygons; ++p){
                if(isInsidePolygon(x + 0.5f, y + 0.5f, &img.polygons[p])){
                    pixel_color = blendColors(pixel_color, img.polygons[p].color);
                    polygon_applied = true;
                }
            }
            float delta_e = calculateDeltaE(pixel_color, target_color);
            local_delta_e += delta_e;
        }
    }
    fitness_scores[idx] = local_delta_e / (width * height);
}

thrust::host_vector<float> calculate_fitness(thrust::host_vector<Image> population, const unsigned char* target_image, int width, int height){
    Image* d_population;
    unsigned char* d_target_image;
    float* d_fitness_scores;

    cudaMalloc(&d_population, POPULATION_SIZE*sizeof(Image));
    cudaMalloc(&d_target_image, width*height*3*sizeof(unsigned char));
    cudaMalloc(&d_fitness_scores, POPULATION_SIZE*sizeof(float));

    cudaMemcpy(d_population, population.data(), POPULATION_SIZE*sizeof(Image), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_image, target_image, width*height*3*sizeof(unsigned char), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (POPULATION_SIZE+blockSize-1)/blockSize;

    fitness_kernel<<<gridSize, blockSize>>>(d_population, d_target_image, d_fitness_scores, width, height);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Fitness Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();

    thrust::device_ptr<float> d_ptr(d_fitness_scores);
    thrust::host_vector<float> h_fitness_scores(d_ptr, d_ptr+POPULATION_SIZE);

    cudaFree(d_population);
    cudaFree(d_target_image);
    cudaFree(d_fitness_scores);

    return h_fitness_scores;
}