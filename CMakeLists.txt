cmake_minimum_required(VERSION 3.22)
project(AlexNet)

set(CMAKE_CXX_STANDARD 14)

add_executable(AlexNet main.cpp Labels.h PixelMatrix.h)
