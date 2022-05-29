//
// Created by andrvat on 19.05.22.
//

#ifndef ALEXNET_HIDDENNEURON_H
#define ALEXNET_HIDDENNEURON_H

#include <vector>
#include <cstdlib>
#include "ImagesContainer.h"
#include "OutputNeuron.h"

enum class NeuronType {
    CONVOLUTION,
    POOLING
};

enum class NextLayerType {
    OUTPUT,
    NOT_OUTPUT,
};

class HiddenNeuron {
private:
    std::vector<std::vector<double>> weights;

    std::pair<int, int> matchPosition;
    double output = 0;

    double localGradient{};

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
                        weights[i][j] = ((double) (AlexNetConstants::LOWER_RANDOM_BORDER +
                                                   rand() % (AlexNetConstants::UPPER_RANDOM_BORDER - AlexNetConstants::LOWER_RANDOM_BORDER + 1))) / AlexNetConstants::SCALE;
                        if (weights[i][j] == 0) {
                            weights[i][j] = AlexNetConstants::DEFAULT_RANDOM_WEIGHT;
                        }
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

    void calcLocalGradient(const std::vector<OutputNeuron> &prevLayerOutputNeurons,
                           const std::vector<HiddenNeuron> &prevLayerHiddenNeurons,
                           const int rowIndex, const int columnIndex,
                           NextLayerType nextLayerType, NeuronType currentNeuronType) {
        double sum = 0;
        switch (currentNeuronType) {
            case NeuronType::CONVOLUTION:
                switch (nextLayerType) {
                    case NextLayerType::NOT_OUTPUT:
                        localGradient = calcDerivativeActivationFunction();
                        for (auto &neuron: prevLayerHiddenNeurons) {
                            if (matchPosition.first <= rowIndex &&
                                matchPosition.first < rowIndex + weights.size() &&
                                matchPosition.second <= columnIndex &&
                                matchPosition.second < columnIndex + weights.size()) {
                                sum += neuron.getLocalGradient() * neuron.getWeights()
                                        .at(rowIndex - matchPosition.first)
                                        .at(columnIndex - matchPosition.second);
                            }
                        }
                        break;
                    case NextLayerType::OUTPUT:
                        localGradient = calcDerivativeActivationFunction();
                        for (auto &neuron: prevLayerOutputNeurons) {
                            sum += neuron.getLocalGradient() * neuron.getWeights().at(rowIndex).at(columnIndex);
                        }
                        break;
                }
                break;
            case NeuronType::POOLING:
                localGradient = output;
                for (auto &neuron: prevLayerHiddenNeurons) {
                    if (matchPosition.first <= rowIndex && matchPosition.first < rowIndex + weights.size() &&
                        matchPosition.second <= columnIndex && matchPosition.second < columnIndex + weights.size()) {
                        sum += neuron.getLocalGradient() * neuron.getWeights()
                                .at(rowIndex - matchPosition.first)
                                .at(columnIndex - matchPosition.second);
                    }
                }
                break;
        }
        localGradient *= sum;
    }

    void updateWeights(const std::vector<std::vector<double>> &image) {
        for (auto i = 0; i < weights.size(); ++i) {
            for (auto j = 0; j < weights.size(); ++j) {
                weights[i][j] += AlexNetConstants::LEARNING_RATE * localGradient *
                                 image[i + matchPosition.first][j + matchPosition.second];
            }
        }
    }

    const std::vector<std::vector<double>> &getWeights() const {
        return weights;
    }

    double getLocalGradient() const {
        return localGradient;
    }
};

#endif //ALEXNET_HIDDENNEURON_H
