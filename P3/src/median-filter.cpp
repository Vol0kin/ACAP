#include <mpi.h>
#include "CImg.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>

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

int main(int argc, char* argv[])
{
    if (argc != 5)
    {
        std::cerr << "Error. Expected one argument" << std::endl;
        std::cerr << "Usage: " << argv[0] << " [path to image] [kernel size] [image width] [image height]" << std::endl;
        return 1;
    }

    int numProcs, myID;
    MPI_Status status;

    // Get parameters
    int kernelSize = atoi(argv[2]);
    int width = atoi(argv[3]);
    int height = atoi(argv[4]);

    int borderSize = kernelSize / 2;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myID);

    int heightPerProcess = height / numProcs;
    int widthBorders = width + 2 * borderSize;

    int divideBlockSize = widthBorders * (heightPerProcess + 2 * borderSize);
    int finalBlockSize = width * heightPerProcess;
    int offset = borderSize * widthBorders + widthBorders * (heightPerProcess - borderSize);

    // Allocate memory for the pixels which every processs will use
    unsigned char* blockPixels = new unsigned char[divideBlockSize];
    std::cout << "Process " << myID << " allocated array of size " << divideBlockSize << " pixels" << std::endl;

    // Process 0 loads image and sends it
    if (myID == 0)
    {
        // Load image
        CImg<unsigned char> image(argv[1]);

        // Use only one of the channels since they're all the same
        image.channel(0);

        // Add borders to image
        CImg<unsigned char> imageWithBorders = addBordersToImage(image, borderSize);

        // Display image information
        std::cout << "Image width: " << imageWithBorders.width() << " Height: " << imageWithBorders.height() << " Depth: " << imageWithBorders.depth() << std::endl;
        // Get image pixels
        unsigned char * pixels = imageWithBorders.data();
        unsigned char * pixelsIter;

        // Copy the part of the image which the process 0 will use
        for (int i = 0; i < divideBlockSize; i++)
        {
            blockPixels[i] = pixels[i];
        }

        pixelsIter = pixels + offset; 

        // Send information to all processes
        for (int dest = 1; dest < numProcs; dest++)
        {
            MPI_Send(pixelsIter, divideBlockSize, MPI_UNSIGNED_CHAR, dest, 0, MPI_COMM_WORLD);
            pixelsIter += offset;
        }
    }
    else
    {
        MPI_Recv(blockPixels, divideBlockSize, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        int recv;
        MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &recv);
        std::cout << "Received " << recv  <<  std::endl;
    }

    // Create image that every process will use
    CImg<unsigned char> processImage(blockPixels, widthBorders, heightPerProcess + 2 * borderSize, 1, 1, false);
    std::cout << "Process " << myID << " width: " <<  processImage.width() << " height: " << processImage.height() << " first: " << (int)processImage(0, 0) << " last-first: " << (int)processImage(0, 129) << std::endl;

    CImgDisplay display(processImage, "Aaaa");

    while (!display.is_closed())
    {
        display.wait();
    }

    MPI_Finalize();

/*



    // Get pixels from image
    unsigned char * pixels = image.data();
    int numPixels = image.size();

    CImg<unsigned char> finalImage = medianFilter(image, kernelSize, borderSize);


    // Display the image (TEST)
    CImgDisplay display(finalImage,  "This is a very cool image");

    while (!display.is_closed())
    {
        display.wait();
    }*/

    return 0;
}
