#include "image_utils.h"
#include <iostream>
#include "CImg.h"

using namespace cimg_library;
using namespace std;

void fillEdgesWithValue(CImg<float>& img, float value, int x0, int y0, int xFinal, int yFinal)
{
    for (int x = x0; x < xFinal; x++)
    {
        for (int y = y0; y < yFinal; y++)
        {
            img(x, y) = value;
        }
    }
}

CImg<float> addBordersToImage(const CImg<float> &image, int borderSize)
{
    // Create new image adding borders to it
    CImg<float> expandedImage(image.width() + 2 * borderSize,
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

vector<float> loadExpandImage(const char* imagePath, int width, int height, int borderSize)
{
    // Load image
    CImg<float> srcImage(imagePath);
    srcImage.channel(0);

    // Expand image and get pixels from it
    CImg<float> expandedImage = addBordersToImage(srcImage, borderSize);

    vector<float> pixels(expandedImage.data(),
                         expandedImage.data() + (width + 2*borderSize) * (height + 2*borderSize));

    return pixels;
}


void saveImage(float* pixels, int width, int height)
{
    CImg<float> image(pixels, width, height, 1, 1, false);
    image.save("output.png");
}
