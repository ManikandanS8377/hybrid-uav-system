#pragma once
#include "Mixer.h"

class Motor {
public:
    void begin();
    void update(Mixer& mixer);
    void failsafe();

private:
    int motorPins[4] = {14, 25, 27, 26}; // AO3400 gate pins
};
