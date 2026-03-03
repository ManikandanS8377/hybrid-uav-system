#pragma once
#include <Arduino.h>

class PID {
public:
    PID(); // default constructor
    PID(float kp, float ki, float kd); // parameterized constructor

    void init(float _kp, float _ki, float _kd);
    float update(float setpoint, float measurement, float dt);
    void reset();
    void setIntegralLimit(float limit);

private:
    float kp, ki, kd;
    float integral;
    float lastError;
    float imax;
};
