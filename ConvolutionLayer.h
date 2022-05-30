#ifndef ALEXNET_CONVOLUTIONLAYER_H
#define ALEXNET_CONVOLUTIONLAYER_H

#include "HiddenNeuron.h"
#include "OutputNeuron.h"
#include <cmath>

class ConvolutionLayer {
private:
    std::vector<HiddenNeuron> neurons;

    std::vector<std::vector<double>> image;

    size_t imageSize;
public:
    void setImageSize(const size_t size) {
        this->imageSize = size;
    }

    void buildLayer(const size_t convolutionSize, const int offset) {
        auto curNeuron = 0;
        for (int i = 0; i + convolutionSize <= imageSize; i += offset) {
            for (int j = 0; j + convolutionSize <= imageSize; j += offset) {
                neurons.emplace_back(convolutionSize, NeuronType::CONVOLUTION);
                neurons.at(curNeuron++).setMatchPosition(std::make_pair(i, j));
            }
        }
    }

    void setInputMatrix(const std::vector<std::vector<double>> &matrix) {
        this->image = matrix;
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

    double applyDeltaRule(const std::vector<OutputNeuron> &prevLayerOutputNeurons,
                          const std::vector<HiddenNeuron> &prevLayerHiddenNeurons,
                          NextLayerType nextLayerType, NeuronType currentNeuronType) {
        for (auto i = 0; i < neurons.size(); ++i) {
            auto &neuron = neurons[i];
            int rowIndex = (int) (i / sqrt((double) neurons.size()));
            int columnIndex = i % (int) sqrt((double) neurons.size());
            neuron.calcLocalGradient(prevLayerOutputNeurons,
                                     prevLayerHiddenNeurons,
                                     rowIndex, columnIndex,
                                     nextLayerType, currentNeuronType);
            neuron.updateWeights(this->image);
        }
    }

    const std::vector<HiddenNeuron> &getNeurons() const {
        return neurons;
    }
};

#endif //ALEXNET_CONVOLUTIONLAYER_H
