#include "image_utils.h"
#include "medianKernel.h"
#include <iostream>

int main(int argc, char* argv[])
{
    if (argc != 7)
    {
        std::cerr << "Error. Expected four argument" << std::endl;
        std::cerr << "Usage: " << argv[0] << " [path to image] [kernel size] [width] "
                     "[height] [window size] [pixels per thread]" << std::endl;
        return 1;
    }

    // Get arguments
    int kernelSize = atoi(argv[2]);
    int width = atoi(argv[3]);
    int height = atoi(argv[4]);
    int windowSize = atoi(argv[5]);
    int pixelsPerThread = atoi(argv[6]);

    // Kernel execution time variable
    double execTime;

    int borderSize = kernelSize / 2;

    std::vector<float> image = loadExpandImage(argv[1], width, height, borderSize);

    float* filteredImage = medianFilter(image.data(), width, height, kernelSize, windowSize,
                                        pixelsPerThread, execTime);

    saveImage(filteredImage, width, height);

    // Show output information
    std::cout << "Pixels per thread (in each dimension),Execution time (seconds)" << std::endl;
    std::cout << pixelsPerThread << "," << execTime << std::endl;

    // Delete output image (it is a dinamically allocated array)
    delete[] filteredImage;

    return 0;
}
