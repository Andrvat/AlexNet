//
// Created by andrvat on 28.05.22.
//

#ifndef ALEXNET_CONSTANTS_H
#define ALEXNET_CONSTANTS_H

namespace AlexNetConstants {
    const auto LEARNING_RATE = 10e-5;
    constexpr auto LOWER_RANDOM_BORDER = -1000;
    constexpr auto UPPER_RANDOM_BORDER = 1000;
    constexpr auto SCALE = 10e3;
    constexpr auto DEFAULT_RANDOM_WEIGHT = 0.006;
    constexpr auto TRAINING_SHARE = 0.7;
}

#endif //ALEXNET_CONSTANTS_H
