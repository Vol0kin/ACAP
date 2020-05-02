#include "image_utils.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>

/*
// Median filter function
CImg<unsigned char> medianFilter(const CImg<unsigned char>& image, int kernelSize, int borderSize)
{
    CImg<unsigned char> filteredImage(image.width() - 2 * borderSize,
                                      image.height() - 2 * borderSize,
                                      image.depth(), 1, 0);
    int anchor = kernelSize / 2;

    // Iterate over image
    for (int y = anchor; y < image.height() - anchor; y++)
    {
        for (int x = anchor; x < image.width() - anchor; x++)
        {
            std::vector<unsigned char> area;

            // Create area around the current pixel which has kernelSize x kernelSize pixels
            for (int j = y - anchor; j <= y + anchor; j++)
            {
                for (int i = x - anchor; i <= x + anchor; i++)
                {
                    area.push_back(image(i, j));
                }
            }

            // Sort values
            std::sort(area.begin(), area.end());

            // Find median value
            unsigned char median = area[area.size() / 2];
            filteredImage(x - anchor, y - anchor) = median;
        }
    }

    return filteredImage;
}
*/
int main(int argc, char* argv[])
{
    if (argc != 5)
    {
        std::cerr << "Error. Expected four argument" << std::endl;
        std::cerr << "Usage: " << argv[0] << " [path to image] [kernel size] [width] [height]" << std::endl;
        return 1;
    }

    // Get kernel size
    int kernelSize = atoi(argv[2]);
    int width = atoi(argv[3]);
    int height = atoi(argv[4]);

    int borderSize = kernelSize / 2;

    // Load image
    float* image = loadImage(argv[1], borderSize);
    saveImage(image, width, height);

    return 0;
}
