#ifndef ALEXNET_ALEXNET_H
#define ALEXNET_ALEXNET_H

#include "ConvolutionLayer.h"
#include "PoolingLayer.h"
#include "OutputLayer.h"

#define DEBUG_OUTPUTS_PRINTING
#define DEBUG_ERRORS_PRINTING

class AlexNet {
private:
    ConvolutionLayer *convolutionLayer_1;
    PoolingLayer *poolingLayer_1;
    ConvolutionLayer *convolutionLayer_2;
    OutputLayer *outputLayer;

    std::vector<double> costFunction;

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

    // TODO: calculate cost function
    void train(ImagesContainer &imagesContainer, const size_t epochNumber) {
        std::cout << "TRAINING.................................." << std::endl;
        std::vector<double> neuralNetworkOutputs;
        size_t currentEpoch = 0;
        bool isNeededBuild = true;
        auto trainImageIndexes = imagesContainer.getTrainingImageIndexes();
        while (currentEpoch < epochNumber) {
            std::cout << "Current epoch: " << currentEpoch << std::endl;
            for (const auto i: trainImageIndexes) {
                if (
                        imagesContainer.getLabelByIndex(i) != 3
//                        && imagesContainer.getLabelByIndex(i) != 7
                        ) {
                    continue;
                }
                const auto &image = imagesContainer.getImageByIndex(i);
                this->makeStraightRunning(image, isNeededBuild);
                neuralNetworkOutputs = outputLayer->getNeuronsOutput();
                std::vector<int> maxProbabilityLabels = AlexNet::calcMaxProbabilityLabels(neuralNetworkOutputs);
#ifdef DEBUG_OUTPUTS_PRINTING
                if (currentEpoch == epochNumber - 1) {
                    std::cout << "FINISHING..." << std::endl;
                }
                printResults(imagesContainer, neuralNetworkOutputs, maxProbabilityLabels, i);
#endif
                auto errorValues = AlexNet::calcErrorValues(imagesContainer.getLabelByIndex(i), neuralNetworkOutputs);
                this->makeBackPropagation(errorValues);
                isNeededBuild = false;
            }
            currentEpoch++;
        }
    }

    void test(ImagesContainer &imagesContainer) {
        std::cout << "TESTING.................................." << std::endl;
        std::vector<double> neuralNetworkOutputs;
        auto testImageIndexes = imagesContainer.getTestImageIndexes();
        for (const auto i: testImageIndexes) {
            const auto &image = imagesContainer.getImageByIndex(i);
            this->makeStraightRunning(image, false);
            neuralNetworkOutputs = outputLayer->getNeuronsOutput();
            std::vector<int> maxProbabilityLabels = AlexNet::calcMaxProbabilityLabels(neuralNetworkOutputs);
            printResults(imagesContainer, neuralNetworkOutputs, maxProbabilityLabels, i);
        }
    }

private:
    static void printResults(ImagesContainer &imagesContainer,
                             const std::vector<double> &neuralNetworkOutputs,
                             const std::vector<int> &maxProbabilityLabels, const size_t i) {
        std::cout << "------------------------------------------" << std::endl;
        std::cout << "Network outputs: " << std::endl;
        for (auto x: neuralNetworkOutputs) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
        std::cout << "Real: " << imagesContainer.getLabelByIndex(i) << std::endl;
        std::cout << "AlexNet: { ";
        for (auto x: maxProbabilityLabels) {
            std::cout << x << " ";
        }
        std::cout << "}" << std::endl;
        std::cout << "------------------------------------------" << std::endl;
    }

    static std::vector<int> calcMaxProbabilityLabels(std::vector<double> &neuralNetworkOutputs) {
        std::vector<int> maxProbabilityLabels;
        double currentMax = 0;
        for (auto j = 0; j < neuralNetworkOutputs.size(); ++j) {
            if (neuralNetworkOutputs[j] > currentMax) {
                currentMax = neuralNetworkOutputs[j];
                maxProbabilityLabels.clear();
                maxProbabilityLabels.push_back(j);
            } else if (neuralNetworkOutputs[j] == currentMax) {
                maxProbabilityLabels.push_back(j);
            }
        }
        return maxProbabilityLabels;
    }

    void makeBackPropagation(std::vector<double> &errors) {
        outputLayer->applyDeltaRule(errors);
        convolutionLayer_2->applyDeltaRule(outputLayer->getNeurons(), std::vector<HiddenNeuron>(),
                                           NextLayerType::OUTPUT, NeuronType::CONVOLUTION);
        poolingLayer_1->applyDeltaRule(convolutionLayer_2->getNeurons());
        convolutionLayer_1->applyDeltaRule(std::vector<OutputNeuron>(), poolingLayer_1->getNeurons(),
                                           NextLayerType::NOT_OUTPUT, NeuronType::CONVOLUTION);
    }

    void makeStraightRunning(const std::vector<std::vector<double>> &image, const bool isNeededBuild) {
        std::vector<std::vector<double>> outputs;
        convolutionLayer_1->setInputMatrix(image);
        if (isNeededBuild) {
            convolutionLayer_1->setImageSize(image.size());
            convolutionLayer_1->buildLayer(5, 3);
        }
        convolutionLayer_1->makeConvolution();

        outputs = convolutionLayer_1->getNeuronsOutput();

        poolingLayer_1->setInputMatrix(outputs);
        if (isNeededBuild) {
            poolingLayer_1->setImageSize(outputs.size());
            poolingLayer_1->buildLayer(2, 1);
        }
        poolingLayer_1->makeAvgPooling();

        outputs = poolingLayer_1->getNeuronsOutput();

        convolutionLayer_2->setInputMatrix(outputs);
        if (isNeededBuild) {
            convolutionLayer_2->setImageSize(outputs.size());
            convolutionLayer_2->buildLayer(3, 1);
        }
        convolutionLayer_2->makeConvolution();

        outputs = convolutionLayer_2->getNeuronsOutput();

        outputLayer->setImage(outputs);
        if (isNeededBuild) {
            outputLayer->setImageSize(outputs.size());
            outputLayer->buildLayer();
        }
        outputLayer->activateNeurons();
    }

    static std::vector<double> calcErrorValues(const int realLabel, const std::vector<double> &neuralNetworkOutputs) {
        std::vector<double> errors(neuralNetworkOutputs.size());
        for (auto i = 0; i < neuralNetworkOutputs.size(); ++i) {
            auto output = neuralNetworkOutputs[i];
            errors[i] = (i == realLabel) ? 1 - output : 0 - output;
        }
#ifdef DEBUG_ERRORS_PRINTING
        std::cout << "***************************************************" << std::endl;
        std::cout << "For real label: " << realLabel << " got errors: ";
        for (auto x: errors) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
        std::cout << "***************************************************" << std::endl;
#endif
        return errors;
    }
};

#endif //ALEXNET_ALEXNET_H
