//
// Created by andrvat on 22.05.22.
//

#ifndef ALEXNET_OUTPUTLAYER_H
#define ALEXNET_OUTPUTLAYER_H

#include <vector>
#include "OutputNeuron.h"

class OutputLayer {
private:
    const int NEURONS_NUMBER = 10;
    std::vector<OutputNeuron> neurons;

public:
    OutputLayer(std::vector<std::vector<double>> &image) {
                neurons.resize(NEURONS_NUMBER);
        for (int i = 0; i < NEURONS_NUMBER; ++i) {

        }

    }
};

#endif //ALEXNET_OUTPUTLAYER_H
