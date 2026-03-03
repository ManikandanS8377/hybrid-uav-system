#pragma once
#include <Arduino.h>
#include "Mixer.h"

class Motor {
public:
    void begin();
    void stopAll();
    void mix(int throttle, float roll, float pitch, float yaw);

    void failsafe();
    void idle();
    void update(const Mixer& mixer);   // ✅ now accepts Mixer reference

private:
    void writeMotor(int channel, int value);
};
