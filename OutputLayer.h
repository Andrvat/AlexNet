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
    std::vector<std::vector<double>> image;

public:
    explicit OutputLayer(std::vector<std::vector<double>> &image) {
        for (int i = 0; i < NEURONS_NUMBER; ++i) {
            neurons.emplace_back(image.size());
        }
        this->image = image;
    }

    void activateNeurons() {
        for (int i = 0; i < NEURONS_NUMBER; ++i) {
            neurons[i].activate(image);
        }
    }


    std::vector<double> getNeuronsOutput() {
        std::vector<double> outputs(NEURONS_NUMBER);
        for (int i = 0; i < NEURONS_NUMBER; ++i) {
            outputs[i] = neurons[i].getOutput();
        }
        return outputs;
    }

};

#endif //ALEXNET_OUTPUTLAYER_H
