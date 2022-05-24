//
// Created by andrvat on 22.05.22.
//

#ifndef ALEXNET_OUTPUTNEURON_H
#define ALEXNET_OUTPUTNEURON_H

#include <vector>
#include <cstdlib>
#include <cmath>

class OutputNeuron {
private:
    const int LOWER_RANDOM_BORDER = 0;
    const int UPPER_RANDOM_BORDER = 1000;
    const int SCALE = 1000;

    std::vector<std::vector<double>> weights;
    double output = 0;

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

    double calcActivationFunction(std::vector<std::vector<double>> &image) {
        auto weightedSum = calcWeightedSum(image);
        auto sigmoid = calcSigmoid(weightedSum);
        return sigmoid;
    }

public:
    explicit OutputNeuron(const size_t size) {
        for (auto i = 0; i < size; ++i) {
            for (auto j = 0; j < size; ++j) {
                weights[i][j] = ((double) (LOWER_RANDOM_BORDER +
                                           rand() % (UPPER_RANDOM_BORDER - LOWER_RANDOM_BORDER + 1))) / SCALE;
            }
        }
    }

    void activate(std::vector<std::vector<double>> &image) {
        output = calcActivationFunction(image);
    }

    double getOutput() const {
        return output;
    }

};

#endif //ALEXNET_OUTPUTNEURON_H
