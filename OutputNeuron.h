//
// Created by andrvat on 22.05.22.
//

#ifndef ALEXNET_OUTPUTNEURON_H
#define ALEXNET_OUTPUTNEURON_H

#include <vector>
#include <cstdlib>
#include <cmath>
#include "Constants.h"

class OutputNeuron {
private:
    std::vector<std::vector<double>> weights;
    double output = 0;

    double localGradient;

    double calcWeightedSum(std::vector<std::vector<double>> &image) {
        double sum = 0;
        for (int i = 0; i < image.size(); ++i) {
            for (int j = 0; j < image.size(); ++j) {
                sum += weights[i][j] * image[i][j];
            }
        }
        return sum;
    }

    static double calcSigmoid(const double s) {
        return 1 / (1 + pow(exp(1), -s));
    }

    static double calcSigmoidDerivative(const double s) {
        return exp(-s) / pow((1 + exp(-s)), 2);
    }

    double calcActivationFunction(std::vector<std::vector<double>> &image) {
        auto weightedSum = calcWeightedSum(image);
        auto sigmoid = calcSigmoid(weightedSum);
        return sigmoid;
    }

public:

    explicit OutputNeuron(const size_t size) {
        weights.resize(size, std::vector<double>(size));
        for (auto i = 0; i < size; ++i) {
            for (auto j = 0; j < size; ++j) {
                weights[i][j] = ((double) (AlexNetConstants::LOWER_RANDOM_BORDER +
                                           rand() % (AlexNetConstants::UPPER_RANDOM_BORDER -
                                                     AlexNetConstants::LOWER_RANDOM_BORDER + 1)))
                                / AlexNetConstants::SCALE;
                if (weights[i][j] == 0) {
                    weights[i][j] = AlexNetConstants::DEFAULT_RANDOM_WEIGHT;
                }
            }
        }
    }

    void activate(std::vector<std::vector<double>> &image) {
        output = calcActivationFunction(image);
    }

    double getOutput() const {
        return output;
    }

    void calcLocalGradient(const double error) {
        localGradient = error * OutputNeuron::calcSigmoidDerivative(output);
    }

    void updateWeights(std::vector<std::vector<double>> &image) {
        for (auto i = 0; i < weights.size(); ++i) {
            for (auto j = 0; j < weights.size(); ++j) {
                weights[i][j] += AlexNetConstants::LEARNING_RATE * localGradient * image[i][j];
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

#endif //ALEXNET_OUTPUTNEURON_H
