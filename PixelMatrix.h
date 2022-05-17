//
// Created by andrvat on 17.05.22.
//

#ifndef ALEXNET_PIXELMATRIX_H
#define ALEXNET_PIXELMATRIX_H

#include <vector>
#include <string>
#include <fstream>

class PixelMatrix {
private:
    std::vector<std::vector<std::vector<double>>> matrices;
public:
    void readFrom(const std::string &filename, const size_t imagesNumber, const size_t matrixSize,
                  const size_t extensionSize) {
        matrices.resize(imagesNumber);
        std::ifstream stream(filename);
        size_t currentImage = 0;
        while (!stream.eof()) {
            matrices.at(currentImage).resize(matrixSize + extensionSize);
            for (auto i = 0; i < matrixSize; ++i) {
                for (auto j = 0; j < matrixSize; ++j) {
                    double x;
                    stream >> x;
                    matrices.at(currentImage).at(i).push_back(x);
                }
                for (auto j = 0; j < extensionSize; ++j) {
                    matrices.at(currentImage).at(i).push_back(0);
                }
            }
            for (auto i = 0; i < extensionSize; ++i) {
                for (auto j = 0; j < matrixSize + extensionSize; ++j) {
                    matrices.at(currentImage).at(matrixSize + i).push_back(0);
                }
            }
            currentImage++;
            if (currentImage == imagesNumber) {
                break;
            }
        }
        stream.close();
    }
};

#endif //ALEXNET_PIXELMATRIX_H
