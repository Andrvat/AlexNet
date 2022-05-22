//
// Created by andrvat on 22.05.22.
//

#ifndef ALEXNET_OUTPUTNEURON_H
#define ALEXNET_OUTPUTNEURON_H

#include <vector>

class OutputNeuron {
private:

    std::vector<std::vector<double>> weights;
    double output = 0;

public:
    OutputNeuron() {

    }
};
#endif //ALEXNET_OUTPUTNEURON_H
