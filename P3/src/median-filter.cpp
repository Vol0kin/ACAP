#include "CImg.h"
#include <iostream>

using namespace cimg_library;

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Error. Expected one argument" << std::endl;
        std::cerr << "Usage: " << argv[0] << " [path to image]" << std::endl;
        return 1;
    }

    // Load image
    CImg<unsigned char> image(argv[1]);

    // Use only one of the channels since they're all the same
    image.channel(0);

    // Display the image (TEST)
    CImgDisplay display(image, "This is a very cool image");

    while (!display.is_closed())
    {
        display.wait();
    }

    return 0;
}
