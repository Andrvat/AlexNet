#ifndef ALEXNET_IMAGESCONTAINER_H
#define ALEXNET_IMAGESCONTAINER_H

#include <vector>
#include <string>
#include <fstream>
#include "Constants.h"

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

    int getLabelByIndex(const size_t i) const {
        return labels.getLabels().at(i);
    }

    size_t getImagesNumber() {
        return matrices.size();
    }

    std::vector<size_t> getTrainingImageIndexes() {
        std::vector<size_t> indexes;
        size_t trainingIndexesSize = matrices.size() * AlexNetConstants::TRAINING_SHARE;
        for (auto i = 0; i < trainingIndexesSize; ++i) {
            indexes.push_back(i);
        }
        return indexes;
    }

    std::vector<size_t> getTestImageIndexes() {
        std::vector<size_t> indexes;
        size_t trainingIndexesSize = matrices.size() * AlexNetConstants::TRAINING_SHARE + 1;
        for (auto i = trainingIndexesSize; i < matrices.size(); ++i) {
            indexes.push_back(i);
        }
        return indexes;
    }

    void setLabels(const Labels &newLabels) {
        this->labels = newLabels;
    }
};

#endif //ALEXNET_IMAGESCONTAINER_H
