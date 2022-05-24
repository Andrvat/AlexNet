//
// Created by andrvat on 19.05.22.
//

#ifndef ALEXNET_ALEXNET_H
#define ALEXNET_ALEXNET_H

#include "ConvolutionLayer.h"
#include "PoolingLayer.h"
#include "OutputLayer.h"

class AlexNet {
private:

    ConvolutionLayer *convolutionLayer_1;
    PoolingLayer *poolingLayer_1;
    ConvolutionLayer *convolutionLayer_2;
    OutputLayer *outputLayer;

public:
    AlexNet() {
        convolutionLayer_1 = new ConvolutionLayer();
        poolingLayer_1 = new PoolingLayer();
        convolutionLayer_2 = new ConvolutionLayer();
        outputLayer = new OutputLayer();
    }

    virtual ~AlexNet() {
        delete convolutionLayer_1;
        delete poolingLayer_1;
        delete convolutionLayer_2;
        delete outputLayer;
    }

    void train(ImagesContainer &imagesContainer) {
        std::vector<std::vector<double>> outputs;
        std::vector<double> neuralNetworkOutputs;
        for (int i = 0; i < imagesContainer.getImagesNumber(); ++i) {
            const auto &image = imagesContainer.getImageByIndex(i);
            convolutionLayer_1->setInputMatrix(image);
            if (i == 0) {
                convolutionLayer_1->setImageSize(image.size());
                convolutionLayer_1->buildLayer(5, 3);
            }
            convolutionLayer_1->makeConvolution();

            outputs = convolutionLayer_1->getNeuronsOutput();

            poolingLayer_1->setInputMatrix(outputs);
            if (i == 0) {
                poolingLayer_1->setImageSize(outputs.size());
                poolingLayer_1->buildLayer(2, 1);
            }
            poolingLayer_1->makeAvgPooling();

            outputs = poolingLayer_1->getNeuronsOutput();

            convolutionLayer_2->setInputMatrix(outputs);
            if (i == 0) {
                convolutionLayer_2->setImageSize(outputs.size());
                convolutionLayer_2->buildLayer(3, 1);
            }
            convolutionLayer_2->makeConvolution();

            outputs = convolutionLayer_2->getNeuronsOutput();

            outputLayer->setImage(outputs);
            if (i == 0) {
                outputLayer->setImageSize(outputs.size());
                outputLayer->buildLayer();
            }
            outputLayer->activateNeurons();

            neuralNetworkOutputs = outputLayer->getNeuronsOutput();
            std::cout << "Network outputs: " << std::endl;
            for (auto x : neuralNetworkOutputs) {
                std::cout << x << " ";
            }
            std::cout << std::endl;
            std::vector<int> maxProbabilityLabels;
            double currentMax = 0;
            for (auto j = 0; j < neuralNetworkOutputs.size(); ++j) {
                if (neuralNetworkOutputs[j] > currentMax) {
                    currentMax = neuralNetworkOutputs[j];
                    maxProbabilityLabels.clear();
                    maxProbabilityLabels.push_back(j + 1);
                } else if (neuralNetworkOutputs[j] == currentMax) {
                    maxProbabilityLabels.push_back(j + 1);
                }
            }
            std::cout << "Real: " << imagesContainer.getLabelByIndex(i) << std::endl;
            std::cout << "AlexNet: ";
            for (auto x : maxProbabilityLabels) {
                std::cout << x << " ";
            }
            std::cout << std::endl;
        }
    }
};

#endif //ALEXNET_ALEXNET_H
