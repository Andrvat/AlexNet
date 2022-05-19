//
// Created by andrvat on 19.05.22.
//

#ifndef ALEXNET_CONVOLUTIONLAYER_H
#define ALEXNET_CONVOLUTIONLAYER_H

#include "Neuron.h"

class ConvolutionLayer {
private:
    std::vector<Neuron> neurons;
public:
    ConvolutionLayer(const std::vector<std::vector<double>> &image, const size_t neuronsNumber,
                     const size_t convolutionSize, const int offset) {
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
};

#endif //ALEXNET_CONVOLUTIONLAYER_H
