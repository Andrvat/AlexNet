cmake_minimum_required(VERSION 3.17)
project(AlexNet)

set(CMAKE_CXX_STANDARD 14)

add_executable(AlexNet main.cpp Labels.h ImagesContainer.h AlexNet.h ConvolutionLayer.h PoolingLayer.h HiddenNeuron.h OutputLayer.h OutputNeuron.h Constants.h)
