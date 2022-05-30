#ifndef ALEXNET_OUTPUTLAYER_H
#define ALEXNET_OUTPUTLAYER_H

#include <vector>
#include "OutputNeuron.h"

class OutputLayer {
private:
    const int NEURONS_NUMBER = 10;

    std::vector<OutputNeuron> neurons;
    std::vector<std::vector<double>> image;

    size_t imageSize;
public:
    void buildLayer() {
        for (int i = 0; i < NEURONS_NUMBER; ++i) {
            neurons.emplace_back(imageSize);
        }
    }

    void setImageSize(size_t size) {
        this->imageSize = size;
    }

    void setImage(const std::vector<std::vector<double>> &matrix) {
        this->image = matrix;
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

    void applyDeltaRule(std::vector<double> &errors) {
        for (auto i = 0; i < neurons.size(); ++i) {
            auto &neuron = neurons[i];
            auto &error = errors[i];
            neuron.calcLocalGradient(error);
            neuron.updateWeights(this->image);
        }
    }

    const std::vector<OutputNeuron> &getNeurons() const {
        return neurons;
    }
};

#endif //ALEXNET_OUTPUTLAYER_H
