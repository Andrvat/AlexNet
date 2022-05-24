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
    const int INIT_IMAGE_SIZE = 29;

    ConvolutionLayer *convolutionLayer_1;
    PoolingLayer poolingLayer_1;
    ConvolutionLayer convolutionLayer_2;
    OutputLayer outputLayer;

public:
    void train(ImagesContainer &imagesContainer) {
        convolutionLayer_1 = new ConvolutionLayer(INIT_IMAGE_SIZE, 5, 3);

        for (int i = 0; i < imagesContainer.getImagesNumber(); ++i) {
            const auto &image = imagesContainer.getImageByIndex(i);

            convolutionLayer_1->setInputMatrix(image);
            convolutionLayer_1->makeConvolution();
            auto convolutionOutputs = convolutionLayer_1->getNeuronsOutput();
        }

    }

};

#endif //ALEXNET_ALEXNET_H
