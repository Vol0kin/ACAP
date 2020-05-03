#include "image_utils.h"
#include "medianKernel.h"
#include <iostream>

int main(int argc, char* argv[])
{
    if (argc != 6)
    {
        std::cerr << "Error. Expected four argument" << std::endl;
        std::cerr << "Usage: " << argv[0] << " [path to image] [kernel size] [width] [height] [window size]" << std::endl;
        return 1;
    }

    // Get kernel size
    int kernelSize = atoi(argv[2]);
    int width = atoi(argv[3]);
    int height = atoi(argv[4]);
    int windowSize = atoi(argv[5]);
    double execTime;

    int borderSize = kernelSize / 2;

    // Load image
    std::vector<float> image = loadExpandImage(argv[1], width, height, borderSize);
    float* filteredImage = medianFilter(image.data(), width, height, kernelSize, windowSize, execTime);
    saveImage(filteredImage, width, height);

    // Show output
    std::cout << "Window size,Execution time (seconds)" << std::endl;
    std::cout << windowSize << "," << execTime << std::endl;

    delete[] filteredImage;

    return 0;
}
