#include "FlightController.h"
#include "Debug.h"

void FlightController::begin() {
    imu.begin();
    motor.begin();
    rc.begin();

    pidRoll.init(1.0, 0.0, 0.0);   // tune values
    pidPitch.init(1.0, 0.0, 0.0);
    pidYaw.init(1.0, 0.0, 0.0);

    Debug::logln("[FlightController] Initialized");
}

void FlightController::update(float dt) {
    uint32_t now = micros();

    // Fixed loop timing
    if (now - lastLoop >= LOOP_US) {
        lastLoop += LOOP_US;
        runFlightLoop(dt);
    }

    // Refresh RC timestamp if valid 
    if (rc.isValid()) { 
        lastRcPacket = millis(); 
    }

    // Failsafe check
    if (millis() - lastRcPacket > FAILSAFE_TIMEOUT) {
        setFailsafe();
    } else if (state == FAILSAFE && rc.isValid()) {
        if (rc.get(ARM_CH) > 1500 && canArm()) {
            state = ARMING;
            Debug::logln("[STATE] RECOVERED → ARMING");
        } else {
            state = DISARMED;
            Debug::logln("[STATE] RECOVERED → DISARMED");
        }
    }

}

void FlightController::runFlightLoop(float dt) {
    imu.update(dt);
    rc.updateConnection();

    if (!rc.isValid()) return;

    int throttle = rc.get(THROTTLE_CH);
    int armSwitch = rc.get(ARM_CH);

    switch (state) {
        case DISARMED:
            if (armSwitch > 1500 && canArm()) {
                state = ARMING;
                Debug::logln("[STATE] ARMING...");
            }
            motor.stopAll();
            break;

        case ARMING:
            static uint32_t armStart = millis();
            if (millis() - armStart > 1000) {
                state = ARMED;
                Debug::logln("[STATE] ARMED");
            }
            motor.stopAll();
            break;

        case ARMED: {
            if (armSwitch < 1500) {
                disarm();
                return;
            }

            float dt = LOOP_US / 1e6;
            float rollCorr  = pidRoll.update(0, imu.getRollRate(), dt);
            float pitchCorr = pidPitch.update(0, imu.getPitchRate(), dt);
            float yawCorr   = pidYaw.update(0, imu.getYawRate(), dt);

            motor.mix(throttle, rollCorr, pitchCorr, yawCorr);
            break;
        }

        case FAILSAFE:
            motor.stopAll();
            Debug::logln("[STATE] FAILSAFE");
            return;
    }
}

bool FlightController::canArm() {
    return rc.get(THROTTLE_CH) < 1050; // throttle low
}

void FlightController::arm() {
    state = ARMING;
}

void FlightController::disarm() {
    state = DISARMED;
    motor.stopAll();
    Debug::logln("[STATE] DISARMED");
}

void FlightController::setFailsafe() {
    state = FAILSAFE;
    motor.stopAll();
}
