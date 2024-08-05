#include "fitness.cuh"

/*
Color Utils taken from https://github.com/ThunderStruct/Color-Utilities/blob/master/ColorUtils.cpp
Modified to work with CUDA (I just added __device__ lmfao)
*/

__device__ ColorUtils::xyzColor rgbToXyz(ColorUtils::rgbColor c){
    float x, y, z, r, g, b;
    
    r = c.r / 255.0; g = c.g / 255.0; b = c.b / 255.0;
    
    if (r > 0.04045)
        r = powf(( (r + 0.055) / 1.055 ), 2.4);
    else r /= 12.92;
    
    if (g > 0.04045)
        g = powf(( (g + 0.055) / 1.055 ), 2.4);
    else g /= 12.92;
    
    if (b > 0.04045)
        b = powf(( (b + 0.055) / 1.055 ), 2.4);
    else b /= 12.92;
    
    r *= 100; g *= 100; b *= 100;
    
    // Calibration for observer @2° with illumination = D65
    x = r * 0.4124 + g * 0.3576 + b * 0.1805;
    y = r * 0.2126 + g * 0.7152 + b * 0.0722;
    z = r * 0.0193 + g * 0.1192 + b * 0.9505;
    
    return ColorUtils::xyzColor(x, y, z);
}

__device__ ColorUtils::CIELABColorSpace xyzToLab(ColorUtils::xyzColor c){
    float x, y, z, l, a, b;
    const float refX = 95.047, refY = 100.0, refZ = 108.883;
    
    // References set at calibration for observer @2° with illumination = D65
    x = c.x / refX; y = c.y / refY; z = c.z / refZ;
    
    if (x > 0.008856)
        x = powf(x, 1 / 3.0);
    else x = (7.787 * x) + (16.0 / 116.0);
    
    if (y > 0.008856)
        y = powf(y, 1 / 3.0);
    else y = (7.787 * y) + (16.0 / 116.0);
    
    if (z > 0.008856)
        z = powf(z, 1 / 3.0);
    else z = (7.787 * z) + (16.0 / 116.0);
    
    l = 116 * y - 16;
    a = 500 * (x - y);
    b = 200 * (y - z);
    
    return ColorUtils::CIELABColorSpace(l, a, b);
}

__device__ float getColorDeltaE(ColorUtils::rgbColor c1, ColorUtils::rgbColor c2) {
    ColorUtils::xyzColor xyzC1 = rgbToXyz(c1), xyzC2 = rgbToXyz(c2);
    ColorUtils::CIELABColorSpace labC1 = xyzToLab(xyzC1), labC2 = xyzToLab(xyzC2);
    
    float deltaE = sqrtf(powf(labC1.l - labC2.l, 2) + powf(labC1.a - labC2.a, 2) + powf(labC1.b - labC2.b, 2));
    
    return deltaE;
}

__global__ void calculate_fitness_kernel(Image* population, Image* original_image, float* fitness_scores) {
    int img_idx = blockIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.z * blockDim.x + threadIdx.x;

    if (img_idx >= POPULATION_SIZE || row >= HEIGHT || col >= WIDTH) return;

    int pixel_idx = row * WIDTH + col;

    unsigned char* pop_data = population[img_idx].data;
    unsigned char* orig_data = original_image->data;

    ColorUtils::rgbColor c1 = {
        pop_data[pixel_idx * 3],
        pop_data[pixel_idx * 3 + 1],
        pop_data[pixel_idx * 3 + 2]
    };

    ColorUtils::rgbColor c2 = {
        orig_data[pixel_idx * 3],
        orig_data[pixel_idx * 3 + 1],
        orig_data[pixel_idx * 3 + 2]
    };

    float deltaE = getColorDeltaE(c1, c2);
    atomicAdd(&fitness_scores[img_idx], deltaE);
}

thrust::host_vector<float> calculate_fitness(thrust::host_vector<Image>& population, Image& original_image) {
    Image* d_population;
    Image* d_original_image;
    float* d_fitness_scores;

    cudaMalloc(&d_population, sizeof(Image) * POPULATION_SIZE);
    cudaMalloc(&d_original_image, sizeof(Image));
    cudaMalloc(&d_fitness_scores, sizeof(float) * POPULATION_SIZE);

    // Allocate and copy image data to create a deep copy for GPU usage
    for (int i = 0; i < POPULATION_SIZE; i++) {
        unsigned char* d_data;
        cudaMalloc(&d_data, WIDTH * HEIGHT * 3 * sizeof(unsigned char));
        cudaMemcpy(d_data, population[i].data, WIDTH * HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
        population[i].data = d_data;
    }

    unsigned char* d_orig_data;
    cudaMalloc(&d_orig_data, WIDTH * HEIGHT * 3 * sizeof(unsigned char));
    cudaMemcpy(d_orig_data, original_image.data, WIDTH * HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    original_image.data = d_orig_data;

    cudaMemcpy(d_population, population.data(), sizeof(Image) * POPULATION_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_original_image, &original_image, sizeof(Image), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);  // 2D thread block
    dim3 gridDim(
        POPULATION_SIZE,
        (original_image.height + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (original_image.width + threadsPerBlock.x - 1) / threadsPerBlock.x
    );

    calculate_fitness_kernel<<<gridDim, threadsPerBlock>>>(d_population, d_original_image, d_fitness_scores);
    
    cudaDeviceSynchronize();

    thrust::host_vector<float> fitness_scores(POPULATION_SIZE);
    cudaMemcpy(fitness_scores.data(), d_fitness_scores, sizeof(float) * POPULATION_SIZE, cudaMemcpyDeviceToHost);

    // Clean up
    for (int i = 0; i < POPULATION_SIZE; i++) {
        cudaFree(population[i].data);
    }
    cudaFree(d_orig_data);
    cudaFree(d_population);
    cudaFree(d_original_image);
    cudaFree(d_fitness_scores);

    return fitness_scores;
}