#pragma once
#include "RcInput.h"

class Mixer {
public:
    // Original update (RC only)
    void update(RcInput& rc);

    // New overload: RC + PID corrections
    void update(RcInput& rc, float rollCorr, float pitchCorr, float yawCorr);

    int outputs[4];  // motor outputs array
};
