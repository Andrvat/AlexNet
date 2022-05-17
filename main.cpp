#include <iostream>
#include "Labels.h"
#include "PixelMatrix.h"

namespace {
    constexpr auto MATRIX_SIZE = 28;
    constexpr auto EXTENSION_SIZE = 1;
}

int main() {
    Labels labels;
    labels.readFrom("../labels.txt");
    labels.printLabels(std::cout);

    PixelMatrix images;
    images.readFrom("../pixels.txt", labels.getLabels().size(),  MATRIX_SIZE, EXTENSION_SIZE);
    return EXIT_SUCCESS;
}
