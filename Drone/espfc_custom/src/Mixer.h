#pragma once
#include <Arduino.h>
#include "RcInput.h"

class Mixer {
public:

    // Update using RC throttle + PID corrections
    void update(RcInput& rc, float rollCorr, float pitchCorr, float yawCorr);

    // Access computed motor outputs
    int getMotor(int index) const;

private:
    int outputs[4]; // store mixed motor values
};
