//
// Created by andrvat on 19.05.22.
//

#ifndef ALEXNET_POOLINGLAYER_H
#define ALEXNET_POOLINGLAYER_H

#include <valarray>
#include "Neuron.h"

class PoolingLayer {
private:
    std::vector<Neuron> neurons;

    std::vector<std::vector<double>> image;
public:
    PoolingLayer(const std::vector<std::vector<double>> &image, const size_t neuronsNumber,
                 const size_t poolingSize, const int offset) {
        this->image = image;
        for (auto i = 0; i < neuronsNumber; ++i) {
            neurons.emplace_back(poolingSize, NeuronType::POOLING);
        }

        auto curNeuron = 0;
        for (int i = 0; i + poolingSize <= image.size(); i += offset) {
            for (int j = 0; j + poolingSize <= image.size(); j += offset) {
                neurons.at(curNeuron++).setMatchPosition(std::make_pair(i, j));
            }
        }
    }

    void makeAvgPooling() {
        for (auto &neuron: neurons) {
            neuron.makeAveragePooing(image);
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
