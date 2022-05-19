//
// Created by andrvat on 19.05.22.
//

#ifndef ALEXNET_CONVOLUTIONLAYER_H
#define ALEXNET_CONVOLUTIONLAYER_H

#include "Neuron.h"
#include <cmath>

class ConvolutionLayer {
private:
    std::vector<Neuron> neurons;

    std::vector<std::vector<double>> image;
public:
    ConvolutionLayer(const std::vector<std::vector<double>> &image, const size_t neuronsNumber,
                     const size_t convolutionSize, const int offset) {
        this->image = image;
        for (auto i = 0; i < neuronsNumber; ++i) {
            neurons.emplace_back(convolutionSize, NeuronType::CONVOLUTION);
        }

        auto curNeuron = 0;
        for (int i = 0; i + convolutionSize <= image.size(); i += offset) {
            for (int j = 0; j + convolutionSize <= image.size(); j += offset) {
                neurons.at(curNeuron++).setMatchPosition(std::make_pair(i, j));
            }
        }
    }

    void makeConvolution() {
        for (auto &neuron: neurons) {
            neuron.makeConvolution(image);
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

#endif //ALEXNET_CONVOLUTIONLAYER_H
