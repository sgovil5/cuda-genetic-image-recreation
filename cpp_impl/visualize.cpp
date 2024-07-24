#include "main.h"

// Helper function to interpolate alpha
unsigned char interpolate_alpha(unsigned char bg, unsigned char fg, unsigned char alpha) {
    return (bg * (255 - alpha) + fg * alpha) / 255;
}

// Function to fill a horizontal scanline
void fill_scanline(unsigned char* buffer, int width, int y, int x1, int x2, const Color& color) {
    if (x1 > x2) std::swap(x1, x2);
    for (int x = x1; x <= x2; ++x) {
        if (x >= 0 && x < width && y >= 0 && y < width) {
            int index = (y * width + x) * 3;
            buffer[index] = interpolate_alpha(buffer[index], color.r, color.a);
            buffer[index + 1] = interpolate_alpha(buffer[index + 1], color.g, color.a);
            buffer[index + 2] = interpolate_alpha(buffer[index + 2], color.b, color.a);
        }
    }
}

// Function to fill a polygon
void fill_polygon(unsigned char* buffer, int width, int height, const Polygon& polygon) {
    std::vector<int> nodes;
    int ymin = height, ymax = 0;

    // Find y range of polygon
    for (const auto& point : polygon.points) {
        ymin = std::min(ymin, point.y);
        ymax = std::max(ymax, point.y);
    }

    // Scanline algorithm
    for (int y = ymin; y <= ymax; ++y) {
        nodes.clear();
        int j = polygon.points.size() - 1;
        for (int i = 0; i < polygon.points.size(); i++) {
            if ((polygon.points[i].y < y && polygon.points[j].y >= y) || 
                (polygon.points[j].y < y && polygon.points[i].y >= y)) {
                nodes.push_back(polygon.points[i].x + 
                    (y - polygon.points[i].y) * (polygon.points[j].x - polygon.points[i].x) / 
                    (polygon.points[j].y - polygon.points[i].y));
            }
            j = i;
        }

        std::sort(nodes.begin(), nodes.end());

        for (size_t i = 0; i < nodes.size(); i += 2) {
            if (i + 1 < nodes.size()) {
                fill_scanline(buffer, width, y, nodes[i], nodes[i + 1], polygon.color);
            }
        }
    }
}

void draw_line(unsigned char* buffer, int width, int height, const Line& line, const Color& color) {
    int x0 = line.p1.x, y0 = line.p1.y;
    int x1 = line.p2.x, y1 = line.p2.y;
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;

    while (true) {
        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
            int index = (y0 * width + x0) * 3;
            buffer[index] = color.r;
            buffer[index + 1] = color.g;
            buffer[index + 2] = color.b;
        }
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

void visualize_image(const Image& img, int width, int height, const char* filename) {
    std::vector<unsigned char> buffer(width * height * 3, 0);

    // Fill background
    for (int i = 0; i < width * height; ++i) {
        buffer[i * 3] = img.background.r;
        buffer[i * 3 + 1] = img.background.g;
        buffer[i * 3 + 2] = img.background.b;
    }

    // Draw and fill polygons
    for (const auto& polygon : img.polygons) {
        fill_polygon(buffer.data(), width, height, polygon);
        
        // Draw outline
        for (const auto& line : polygon.lines) {
            draw_line(buffer.data(), width, height, line, polygon.color);
        }
    }

    // Save image
    stbi_write_png(filename, width, height, 3, buffer.data(), width * 3);
}