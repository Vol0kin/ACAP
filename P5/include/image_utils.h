#ifndef _IMAGE_UTILS_H_
#define _IMAGE_UTILS_H_

#include <vector>

/**
 * Function to load and expand an image.
 */ 
std::vector<float> loadExpandImage(const char* imagePath, int width, int height, int borderSize);

/**
 * Function that saves an image
 */ 
void saveImage(float* pixels, int width, int height);

#endif
