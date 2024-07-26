#ifndef TYPES_H
#define TYPES_H

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#define MAX_POINTS 15
#define MAX_POLYGONS 150

struct Color {
    unsigned long long r, g, b;
    unsigned long long a;
};

struct Point {
    unsigned int x, y;
};

struct Line {
    Point p1, p2;
};

struct Polygon {
    Color color;
    Point points[MAX_POINTS];
    Line lines[MAX_POINTS];
    int num_points;
};

struct Image {
    Color background;
    Polygon polygons[MAX_POLYGONS];
    int num_polygons;
};

#endif // TYPES_H