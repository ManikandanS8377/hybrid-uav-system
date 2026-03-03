#include "Motor.h"
#include "Config.h"
#include "Debug.h"
#include <Arduino.h>

void Motor::begin() {
    // Configure LEDC PWM channels for each motor pin
    for (int i = 0; i < 4; i++) {
        ledcSetup(i, 20000, 10);          // channel=i, freq=20kHz, resolution=10-bit
        ledcAttachPin(motorPins[i], i);   // attach pin to channel
    }
    Debug::logln("[Motor] PWM channels initialized (AO3400 drivers)");
}

void Motor::update(Mixer& mixer) {
    for (int i = 0; i < 4; i++) {
        // Map mixer outputs (1000–2000) to PWM duty (0–1023)
        int pwmValue = map(mixer.outputs[i], MOTOR_MIN, MOTOR_MAX, 0, 1023);
        ledcWrite(i, pwmValue);  // drive AO3400 gate with PWM

        Debug::log("[Motor] Pin "); Debug::log(motorPins[i]);
        Debug::log(" -> PWM="); Debug::logln(pwmValue);
    }
}

void Motor::failsafe() {
    for (int i = 0; i < 4; i++) {
        ledcWrite(i, 0);  // cut motors immediately
    }
    Debug::logln("[Motor] FAILSAFE: Motors off");
}
