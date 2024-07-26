#ifndef MAIN_H
#define MAIN_H

#define _USE_MATH_DEFINES

#include "../stb/stb_image.h"
#include "../stb/stb_image_write.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>

struct Color {
    int r, g, b, a;
};

struct Point {
    int x, y;
};

struct Line{
    Point p1, p2;
};

struct Polygon {
    Color color;
    std::vector<Point> points;
    std::vector<Line> lines;
};

struct Image {
    Color background;
    std::vector<Polygon> polygons;
};

void visualize_image(const Image& img, int width, int height, const char* filename);

#endif