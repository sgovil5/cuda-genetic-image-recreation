#include "main.h"

std::random_device rd;
unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::mt19937 gen(seed);
std::uniform_real_distribution<> uniform_dis(0.0, 1.0);
std::uniform_int_distribution<> color_dis(0, 255);
std::uniform_int_distribution<> num_polygons_dis(3, 7);
std::uniform_int_distribution<> num_points_dis(3, 6);


Color avg_color(unsigned char* image, int width, int height, int channels) {
    long long total_r = 0, total_g = 0, total_b = 0;
    int total_pixels = width * height;

    for (int i = 0; i < total_pixels * channels; i += channels) {
        total_r += image[i];
        total_g += image[i + 1];
        total_b += image[i + 2];
    }

    return Color{
        static_cast<int>(total_r / total_pixels),
        static_cast<int>(total_g / total_pixels),
        static_cast<int>(total_b / total_pixels),
        255
    };
}

std::vector<Polygon> generate_polygons(int width, int height){
    // distributions for point generation
    std::uniform_int_distribution<> x_dis(0, width-1);
    std::uniform_int_distribution<> y_dis(0, height-1);

    std::vector<Polygon> polygons;
    int num_polygons = num_polygons_dis(gen);
    for(int i=0; i<num_polygons; i++){
        Polygon polygon;

        // generate color
        polygon.color = {color_dis(gen), color_dis(gen), color_dis(gen), color_dis(gen)};

        // genereate points
        int num_points = num_points_dis(gen);
        for(int j=0; j<num_points; j++){
            Point p = {x_dis(gen), y_dis(gen)};
            polygon.points.push_back(p);
        }

        // generate lines
        for(int j=0; j<num_points; j++){
            Line l = {polygon.points[j], polygon.points[(j+1)%num_points]};
            polygon.lines.push_back(l);
        }

        polygons.push_back(polygon);
    }
    return polygons;
}

Image generate_image(int width, int height, Color avg_color) {
    Image img;
    
    //randomly set background
    double background_prob = uniform_dis(gen);
    if(background_prob<0.4) img.background = avg_color;
    else if(background_prob<0.6) img.background = {0, 0, 0, 255}; //black
    else if(background_prob<0.8) img.background = {255, 255, 255, 255}; //white
    else{
        img.background = {color_dis(gen), color_dis(gen), color_dis(gen), 255};
    }

    //randomly assign polygons
    img.polygons = generate_polygons(width, height);

    return img;
}

std::vector<Image> init_population(int population_size, int width, int height, Color avg_color){
    std::vector<Image> population;
    for(int i=0; i<population_size; i++){
        population.push_back(generate_image(width, height, avg_color));
    }
    return population;
}

int main() {
    int width, height, channels;
    std::string path = "../image.png";
    unsigned char* image = stbi_load(path.c_str(), &width, &height, &channels, 3);

    if (!image) {
        std::cerr << "Error: Could not load image." << std::endl;
        return 1;
    }

    Color averageColor = avg_color(image, width, height, 3);
    
    stbi_image_free(image);

    std::vector<Image> population = init_population(1, width, height, averageColor);
    Image img = population[0];
    visualize_image(img, width, height, "i1.png");
    return 0;
}