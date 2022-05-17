#include <iostream>
#include "Labels.h"
#include "ImagesContainer.h"

namespace {
    constexpr auto MATRIX_SIZE = 28;
    constexpr auto EXTENSION_SIZE = 1;
}

int main() {
    Labels labels;
    labels.readFrom("../labels.txt");
    labels.printLabels(std::cout);

    ImagesContainer images;
    images.readFrom("../pixels.txt", labels.getLabels().size(),  MATRIX_SIZE, EXTENSION_SIZE);
    return EXIT_SUCCESS;
}
