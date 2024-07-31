#ifndef TYPES_H
#define TYPES_H

#define MAX_POINTS 15
#define MAX_POLYGONS 150

#define POPULATION_SIZE 5
#define TOURNAMENT_SIZE 2

struct Color {
    float r, g, b, a;
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

// exists only because of compilation time when passing Image to blendParents?!?!
struct CrossoverImage {
    Color background;
    int num_polygons;
    Polygon* polygons;
};

struct Image {
    Color background;
    Polygon polygons[MAX_POLYGONS];
    int num_polygons;
    float fitness_score;
};

#endif // TYPES_H