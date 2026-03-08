#include "Motor.h"
#include "Config.h"
#include "Debug.h"

void Motor::begin() {
    for (int i = 0; i < 4; i++) {
        ledcSetup(i, 50, 16); // 50 Hz, 16-bit resolution
        ledcAttachPin(MOTOR_PINS[i], i);
    }
    stopAll();
}

void Motor::stopAll() {
    for (int i = 0; i < 4; i++) {
        writeMotor(i, MOTOR_MIN); // full stop
    }
    Debug::logln("[Motor] All motors stopped");
    digitalWrite(STATUS_LED_PIN, LOW);
}

void Motor::failsafe() {
    stopAll();
    Debug::logln("[Motor] FAILSAFE triggered");
}

void Motor::idle() {
    for (int i = 0; i < 4; i++) {
        writeMotor(i, MOTOR_IDLE);
    }
    Debug::logln("[Motor] Motors set to idle");
}

// Only one definition of update()
void Motor::update(const Mixer& mixer) {
    for (int i = 0; i < 4; i++) {
        int value = constrain(mixer.getMotor(i), MOTOR_MIN, MOTOR_MAX);
        writeMotor(i, value);
    }
}

void Motor::writeMotor(int channel, int value) {
    int duty = map(value, MOTOR_MIN, MOTOR_MAX, 0, 65535); 
    ledcWrite(channel, duty);
}
