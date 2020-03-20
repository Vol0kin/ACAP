#include "CImg.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
//#include "mpi.h"

using namespace cimg_library;

// Function to fill edges of the expanded image with a given value
void fillEdgesWithValue(CImg<unsigned char>& img,
                        unsigned char value,
                        int x0,
                        int y0,
                        int xFinal,
                        int yFinal)
{
    for (int x = x0; x < xFinal; x++)
    {
        for (int y = y0; y < yFinal; y++)
        {
            img(x, y) = value;
        }
    }
}

// Function to replicate the borders of a given image
CImg<unsigned char> addBordersToImage(const CImg<unsigned char>& image, int borderSize)
{
    // Create new image adding borders to it
    CImg<unsigned char> expandedImage(image.width() + 2 * borderSize,
                                      image.height() + 2 * borderSize,
                                      image.depth(), 1, 0);

    // Copy original content in the corresponding positions
    for (int x = 0; x < image.width(); x++)
    {
        for (int y = 0; y < image.height(); y++)
        {
            expandedImage(x + borderSize, y + borderSize) = image(x, y);
        }
    }

    // Expand right border
    for (int x = 0; x < borderSize; x++)
    {
        for (int y = borderSize; y < image.height() + borderSize; y++)
        {
            expandedImage(x, y) = expandedImage(borderSize, y);
        }
    }

    // Expand top border
    for (int x = borderSize; x < image.width() + borderSize; x++)
    {
        for (int y = 0; y < borderSize; y++)
        {
            expandedImage(x, y) = expandedImage(x, borderSize);
        }
    }

    // Expand left border
    for (int x = image.width() + borderSize; x < expandedImage.width(); x++)
    {
        for (int y = borderSize; y < image.height() + borderSize; y++)
        {
            expandedImage(x, y) = expandedImage(image.width() + borderSize - 1, y);
        }
    }

    // Expand bottom border
    for (int x = borderSize; x < image.width() + borderSize; x++)
    {
        for (int y = image.height() + borderSize; y < expandedImage.height(); y++)
        {
            expandedImage(x, y) = expandedImage(x, image.height() + borderSize - 1);
        }
    }

    // Fill edges
    fillEdgesWithValue(expandedImage, image(0,0), 0, 0, borderSize, borderSize); // Top-left edge

    fillEdgesWithValue(expandedImage, image(0, image.height() - 1), 0,
                       image.height() + borderSize, borderSize,
                       expandedImage.height());                                  // Bottom-left edge

    fillEdgesWithValue(expandedImage, image(image.width() - 1, 0),
                       image.width() + borderSize, 0, expandedImage.width(),
                       borderSize);                                              // Top-right edge

    fillEdgesWithValue(expandedImage, image(image.width() - 1, image.height() - 1),
                       image.width() + borderSize, image.height() + borderSize,
                       expandedImage.width(), expandedImage.height());           // Bottom-right edge



    return expandedImage;
}

// Median filter function
std::vector<std::vector<unsigned char>> medianFilter(const std::vector<std::vector<unsigned char>>& image,
                                                     int kernelSize)
{
    std::vector<std::vector<unsigned char>> filteredImage;
    int anchor = kernelSize / 2;

    // Iterate over image
    for (int y = anchor; y < image.size() - anchor; y++)
    {
        std::vector<unsigned char> row;

        for (int x = anchor; x < image[0].size() - anchor; x++)
        {
            std::vector<unsigned char> area;

            // Create area around the current pixel which has kernelSize x kernelSize pixels
            for (int i = y - anchor; i <= y + anchor; i++)
            {
                for (int j = x - anchor; j <= x + anchor; j++)
                {
                    area.push_back(image[i][j]);
                }
            }

            // Sort values
            std::sort(area.begin(), area.end());

            // Find median value
            unsigned char median = area[area.size() / 2];
            row.push_back(median);
        }

        // Add row to the final image
        filteredImage.push_back(row);
    }

    return filteredImage;
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Error. Expected one argument" << std::endl;
        std::cerr << "Usage: " << argv[0] << " [path to image] [kernel size]" << std::endl;
        return 1;
    }

    // Get kernel size
    int kernelSize = atoi(argv[2]);

    // Load image
    CImg<unsigned char> image(argv[1]);

    // Use only one of the channels since they're all the same
    image.channel(0);

    // Display image information
    std::cout << "Image width: " << image.width() << " Height: " << image.height() << " Depth: " << image.depth() << std::endl;

    // Add borders to image
    int borderSize = kernelSize / 2;
    image = addBordersToImage(image, borderSize);

    // Get pixels from image
    unsigned char * pixels = image.data();
    int numPixels = image.size();

    // Put data into a matrix
    std::vector<std::vector<unsigned char>> imageData;

    for (int i = 0; i < image.height(); i++)
    {
        std::vector<unsigned char> row;
        row.insert(row.end(), &pixels[i * image.width()], &pixels[(i+1) * image.width()]);
        imageData.push_back(row);
    }

    std::vector<std::vector<unsigned char>> correctedImage = medianFilter(imageData, kernelSize);


    // Create corrected image
    CImg<unsigned char> finalImage(correctedImage[0].size(), correctedImage.size(), 1, 1, 0);

    for (int y = 0; y < correctedImage.size(); y++)
    {
        for (int x = 0; x < correctedImage[0].size(); x++)
        {
            finalImage(x, y) = correctedImage[y][x];
        }
    }


    // Display the image (TEST)
    CImgDisplay display(finalImage,  "This is a very cool image");

    while (!display.is_closed())
    {
        display.wait();
    }

    return 0;
}
