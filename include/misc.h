#ifndef MISC_H
#define MISC_H

#include "platform.h"
#include <random>
#include <ctime>

namespace random {
    std::default_random_engine engine;

    void init_generator() {
        engine.seed(time(0));
    }

    float normal(float mean, float stddev) {
        std::normal_distribution<float> dist(mean, stddev);
        return dist(engine);
    }

    float uniform(float min, float max) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(engine);
    }
}


#endif // MISC_H
