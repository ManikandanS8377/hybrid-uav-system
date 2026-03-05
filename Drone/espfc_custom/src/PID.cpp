#include "PID.h"

PID::PID() : kp(0), ki(0), kd(0), integral(0), lastError(0), imax(100.0f) {}

PID::PID(float _kp, float _ki, float _kd)
    : kp(_kp), ki(_ki), kd(_kd), integral(0), lastError(0), imax(100.0f) {}

void PID::init(float _kp, float _ki, float _kd) {
    kp = _kp;
    ki = _ki;
    kd = _kd;
    integral = 0;
    lastError = 0;
}

float PID::update(float setpoint, float measurement, float dt) {
    // Error
    float error = setpoint - measurement;

    // Integral with windup protection
    integral += error * dt;
    integral = constrain(integral, -imax, imax);

    // Derivative normalized by dt (avoid division by zero)
    float derivative = 0.0f;
    if (dt > 0.0f) {
        derivative = (error - lastError) / dt;
    }
    lastError = error;

    // PID output
    return (kp * error) + (ki * integral) + (kd * derivative);
}

void PID::reset() {
    integral = 0;
    lastError = 0;
}

void PID::setIntegralLimit(float limit) {
    imax = limit;
}
