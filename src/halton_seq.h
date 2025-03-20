#pragma once

#include <vector>
#include <math.h>

// Function to generate the nth value of the Halton sequence for base b
double halton(int index, int base) {
    double result = 0;
    double f = 1.0 / base;
    while (index > 0) {
        result += f * (index % base);
        index /= base;
        f /= base;
    }
    return result;
}

// Function to sample neighbors using a low-discrepancy sequence
std::vector<std::pair<int, int>> sample_low_discrepancy_neighbors(int num_samples, int x, int y, int radius, int w, int h) {
    std::vector<std::pair<int, int>> neighbors;
    
    for (int i = 1; i <= num_samples; ++i) {
        double u = halton(i, 2); // Halton sequence with base 2
        double v = halton(i, 3); // Halton sequence with base 3
        
        // Convert Halton sequence values to polar coordinates
        double r = radius * sqrt(u);
        double theta = 2 * M_PI * v;
        
        // Compute new neighbor coordinates
        int x_new = static_cast<int>(round(x + r * cos(theta)));
        int y_new = static_cast<int>(round(y + r * sin(theta)));
        
        // Clip to image boundaries
        x_new = std::clamp(x_new, 0, w - 1);
        y_new = std::clamp(y_new, 0, h - 1);
        
        neighbors.emplace_back(x_new, y_new);
    }
    
    return neighbors;
}