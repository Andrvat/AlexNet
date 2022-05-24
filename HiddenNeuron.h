//
// Created by andrvat on 19.05.22.
//

#ifndef ALEXNET_HIDDENNEURON_H
#define ALEXNET_HIDDENNEURON_H

#include <vector>
#include <cstdlib>
#include "ImagesContainer.h"

enum class NeuronType {
    CONVOLUTION,
    POOLING
};

class HiddenNeuron {
private:
    const int LOWER_RANDOM_BORDER = 0;
    const int UPPER_RANDOM_BORDER = 1000;
    const int SCALE = 10e4;

    std::vector<std::vector<double>> weights;

    std::pair<int, int> matchPosition;
    double output = 0;

    double static calcActivationFunction(const double s) {
        return std::max(0.0, s);
    }

    double static calcDerivativeActivationFunction() {
        return 1.0;
    }

public:
    HiddenNeuron(const size_t size, const NeuronType type) {
        weights.resize(size, std::vector<double>(size));
        switch (type) {
            case NeuronType::CONVOLUTION:
                for (auto i = 0; i < size; ++i) {
                    for (auto j = 0; j < size; ++j) {
                        weights[i][j] = ((double) (LOWER_RANDOM_BORDER +
                                                   rand() % (UPPER_RANDOM_BORDER - LOWER_RANDOM_BORDER + 1))) / SCALE;
                    }
                }
                break;
            case NeuronType::POOLING:
                for (auto i = 0; i < size; ++i) {
                    for (auto j = 0; j < size; ++j) {
                        weights[i][j] = 1.0;
                    }
                }
                break;
        }
    }

    void setMatchPosition(const std::pair<int, int> &position) {
        matchPosition = position;
    }

    void makeConvolution(const std::vector<std::vector<double>> &image) {
        auto convolutionSize = weights.size();
        double weightedSum = 0;
        for (auto i = 0; i < convolutionSize; ++i) {
            for (auto j = 0; j < convolutionSize; ++j) {
                weightedSum += image[matchPosition.first + i][matchPosition.second + j] * weights[i][j];
            }
        }
        output = calcActivationFunction(weightedSum);
    }

    void makeAveragePooling(const std::vector<std::vector<double>> &image) {
        auto poolingSize = weights.size();
        double sum = 0;
        for (auto i = 0; i < poolingSize; ++i) {
            for (auto j = 0; j < poolingSize; ++j) {
                sum += image[matchPosition.first + i][matchPosition.second + j];
            }
        }
        output = sum / (double) (poolingSize * poolingSize);
    }

    double getOutput() const {
        return output;
    }
};

#endif //ALEXNET_HIDDENNEURON_H
