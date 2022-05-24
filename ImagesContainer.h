//
// Created by andrvat on 17.05.22.
//

#ifndef ALEXNET_IMAGESCONTAINER_H
#define ALEXNET_IMAGESCONTAINER_H

#include <vector>
#include <string>
#include <fstream>

class ImagesContainer {
private:
    std::vector<std::vector<std::vector<double>>> matrices;
    Labels labels;
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


    const std::vector<std::vector<double>> &getImageByIndex(const size_t i) const {
        return matrices.at(i);
    }

    size_t getImagesNumber () {
        return matrices.size();
    }

    void setLabels(const Labels &labels) {
        this->labels = labels;
    }
};

#endif //ALEXNET_IMAGESCONTAINER_H
