#include "FlightController.h"
#include "Debug.h"

extern unsigned long LOOP_PERIOD;

uint32_t armStartTime = 0;

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
    // Check RC connection
    runFlightLoop(dt);
}

void FlightController::runFlightLoop(float dt) {
    imu.update(dt);
    rc.updateConnection();

    if (!rc.isValid()) {
        if (state != FAILSAFE) {
            setFailsafe();
        }
        return;
    }

    int throttle = rc.get(THROTTLE_CH);
    int armSwitch = rc.get(ARM_CH);

    switch (state) {
        case DISARMED:
            if (armSwitch > 1500 && canArm()) {
                state = ARMING;
                armStartTime = millis();
                Debug::logln("[STATE] ARMING...");
            }
            motor.stopAll();
            break;

        case ARMING:
            if (millis() - armStartTime > 1000) {
                state = ARMED;
                Debug::logln("[STATE] ARMED");
            }
            motor.stopAll();
            break;

        case ARMED: {
            if (armSwitch < 1500) {
                disarm();
                digitalWrite(STATUS_LED_PIN, LOW);
                return;
            }

            digitalWrite(STATUS_LED_PIN, HIGH);
            
            float dt = LOOP_PERIOD / 1e6;

            // --- Get raw stick values ---
            int rollInput  = rc.get(ROLL_CH);
            int pitchInput = rc.get(PITCH_CH);
            int yawInput   = rc.get(YAW_CH);

            // --- Apply deadband (±15 around center 1500) ---
            if (abs(rollInput - 1500) < 15)  rollInput  = 1500;
            if (abs(pitchInput - 1500) < 15) pitchInput = 1500;
            if (abs(yawInput - 1500) < 15)   yawInput   = 1500;

            // --- Convert to rate setpoints ---
            float rollSetpoint  = (rollInput  - 1500) * 0.2f;
            float pitchSetpoint = (pitchInput - 1500) * 0.2f;
            float yawSetpoint   = (yawInput   - 1500) * 0.2f;

            // PID
            float rollCorr  = pidRoll.update(rollSetpoint,  imu.getRollRate(), dt);
            float pitchCorr = pidPitch.update(pitchSetpoint, imu.getPitchRate(), dt);
            float yawCorr   = pidYaw.update(yawSetpoint,    imu.getYawRate(), dt);

            // --- Add PID Output Limit ---
            rollCorr  = constrain(rollCorr,  -300, 300);
            pitchCorr = constrain(pitchCorr, -300, 300);
            yawCorr   = constrain(yawCorr,   -300, 300);

            motor.mix(throttle, rollCorr, pitchCorr, yawCorr);
            break;
        }

        case FAILSAFE:
            digitalWrite(STATUS_LED_PIN, LOW);
            motor.stopAll();
            if (rc.isValid()) {
                state = DISARMED;
                Debug::logln("[STATE] RECOVERED → DISARMED");
            }
            return;
    }
}

bool FlightController::canArm() {
    return rc.get(THROTTLE_CH) <= MOTOR_MIN + 100; // throttle low
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
    if (state != FAILSAFE) {
        state = FAILSAFE;
        motor.stopAll();
        Debug::logln("[STATE] FAILSAFE (RC Lost)");
    }
}
