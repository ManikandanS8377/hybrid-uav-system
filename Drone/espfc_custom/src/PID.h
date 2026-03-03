#pragma once
#include <Arduino.h>

class PID {
public:
    PID(float kp, float ki, float kd);

    float update(float setpoint, float measured, float dt);

    void reset();

private:
    float kp, ki, kd;
    float integral;
    float lastError;
};
