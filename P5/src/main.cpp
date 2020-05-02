#include "image_utils.h"
#include "medianKernel.h"
#include <iostream>

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
    std::vector<float> image = loadExpandImage(argv[1], width, height, borderSize);
    medianFilter(image.data(), width, height, kernelSize, 16);
    saveImage(image.data(), width + 2*borderSize, height + 2*borderSize);

    return 0;
}
