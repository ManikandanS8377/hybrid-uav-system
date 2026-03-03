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

void Motor::mix(int throttle, float roll, float pitch, float yaw) {
    // QUADX mixer math
    int m0 = throttle + roll - pitch + yaw; // front right
    int m1 = throttle - roll - pitch - yaw; // front left
    int m2 = throttle - roll + pitch + yaw; // rear left
    int m3 = throttle + roll + pitch - yaw; // rear right

    // Clamp outputs
    m0 = constrain(m0, MOTOR_MIN, MOTOR_MAX);
    m1 = constrain(m1, MOTOR_MIN, MOTOR_MAX);
    m2 = constrain(m2, MOTOR_MIN, MOTOR_MAX);
    m3 = constrain(m3, MOTOR_MIN, MOTOR_MAX);

    writeMotor(0, m0);
    writeMotor(1, m1);
    writeMotor(2, m2);
    writeMotor(3, m3);
}

// ✅ Only one definition of update()
void Motor::update(const Mixer& mixer) {
    for (int i = 0; i < 4; i++) {
        int value = constrain(mixer.getMotor(i), MOTOR_MIN, MOTOR_MAX);
        writeMotor(i, value);
    }
}

void Motor::writeMotor(int channel, int value) {
    ledcWrite(channel, value);
}
