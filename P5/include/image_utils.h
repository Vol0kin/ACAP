#ifndef _IMAGE_UTILS_H_
#define _IMAGE_UTILS_H_

#include "CImg.h"
#include <vector>

/**
 * Function to load and expand an image.
 */ 
std::vector<float> loadExpandImage(const char* imagePath, int width, int height, int borderSize);

/**
 * Function that fills the edges of the expanded image with a given value.
 */ 
void fillEdgesWithValue(cimg_library::CImg<float>& img, float value, int x0,
                        int y0, int xFinal, int yFinal);
/**
 * Function that adds borders to a given image
 */ 
cimg_library::CImg<float> addBordersToImage(const cimg_library::CImg<float> &image,
                                            int borderSize);

/**
 * Function that saves an image
 */ 
void saveImage(float* pixels, int width, int height);

#endif
