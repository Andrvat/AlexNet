//
// Created by andrvat on 19.05.22.
//

#ifndef ALEXNET_POOLINGLAYER_H
#define ALEXNET_POOLINGLAYER_H

#include <valarray>
#include "HiddenNeuron.h"

class PoolingLayer {
private:
    std::vector<HiddenNeuron> neurons;

    std::vector<std::vector<double>> image;

    size_t imageSize;
public:
    void setImageSize(const size_t size) {
        this->imageSize = size;
    }

    void buildLayer(const size_t poolingSize, const int offset) {
        auto curNeuron = 0;
        for (int i = 0; i + poolingSize <= imageSize; i += offset) {
            for (int j = 0; j + poolingSize <= imageSize; j += offset) {
                neurons.emplace_back(poolingSize, NeuronType::POOLING);
                neurons.at(curNeuron++).setMatchPosition(std::make_pair(i, j));
            }
        }
    }

    void setInputMatrix(const std::vector<std::vector<double>> &matrix) {
        this->image = matrix;
    }

    void makeAvgPooling() {
        for (auto &neuron: neurons) {
            neuron.makeAveragePooling(image);
        }
    }

    std::vector<std::vector<double>> getNeuronsOutput() {
        auto matrixOutputSize = (size_t) sqrt((double) neurons.size());
        std::vector<std::vector<double>> outputs(matrixOutputSize, std::vector<double>(matrixOutputSize));
        size_t rowIndex = 0;
        size_t columnIndex = 0;
        for (auto &neuron: neurons) {
            outputs[rowIndex][columnIndex] = neuron.getOutput();
            columnIndex++;
            if (columnIndex % matrixOutputSize == 0) {
                rowIndex++;
                columnIndex = 0;
                if (rowIndex % matrixOutputSize == 0) {
                    break;
                }
            }
        }
        return outputs;
    }
};

#endif //ALEXNET_POOLINGLAYER_H
