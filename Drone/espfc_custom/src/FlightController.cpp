#include "FlightController.h"
#include "Debug.h"

extern unsigned long LOOP_PERIOD;

uint32_t armStartTime = 0;

void FlightController::begin() {

    imu.begin();
    motor.begin();
    rc.begin();

    // Rate PID (used after angle loop)
    pidRoll.init(0.5, 0.0, 0.02);
    pidPitch.init(0.5, 0.0, 0.02);
    pidYaw.init(0.5, 0.0, 0.0);

    Debug::logln("[FlightController] Initialized");
}

void FlightController::update(float dt) {
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

    int armSwitch = rc.get(ARM_CH);

    bool armHigh = (armSwitch > 1700);
    bool armLow  = (armSwitch < 1300);

    switch (state) {

        case DISARMED:

            if (requireArmLow) {
                if (armLow) {
                    requireArmLow = false;
                    Debug::logln("[STATE] ARM switch cleared, ready to arm");
                }
                motor.stopAll();
                break;
            }

            if (armHigh && canArm()) {
                state = ARMING;
                armStartTime = millis();
                Debug::logln("[STATE] ARMING...");
            }

            motor.stopAll();
            break;


        case ARMING:

            motor.idle();

            if (millis() - armStartTime > 1000) {

                pidRoll.reset();
                pidPitch.reset();
                pidYaw.reset();

                wasArmed = true;
                state = ARMED;

                Debug::logln("[STATE] ARMED");
            }

            break;


        case ARMED: {

            if (armLow) {
                disarm();
                digitalWrite(STATUS_LED_PIN, LOW);
                return;
            }

            digitalWrite(STATUS_LED_PIN, HIGH);

            int rollInput  = rc.get(ROLL_CH);
            int pitchInput = rc.get(PITCH_CH);
            int yawInput   = rc.get(YAW_CH);

            // deadband
            if (abs(rollInput  - 1500) < 20) rollInput  = 1500;
            if (abs(pitchInput - 1500) < 20) pitchInput = 1500;
            if (abs(yawInput   - 1500) < 20) yawInput   = 1500;


            // -------------------------
            // ANGLE STABILIZATION LOOP
            // -------------------------

            float rollAngleTarget  = (rollInput  - 1500) * 0.05f;
            float pitchAngleTarget = (pitchInput - 1500) * 0.05f;

            float rollAngle  = imu.getRollAngle();
            float pitchAngle = imu.getPitchAngle();

            // corrected sign (important!)
            float rollAngleError  = rollAngle - rollAngleTarget;
            float pitchAngleError = pitchAngle - pitchAngleTarget;

            // convert angle error to desired rotation rate
            float rollRateTarget  = rollAngleError * 5.0f;
            float pitchRateTarget = pitchAngleError * 5.0f;


            // -------------------------
            // RATE PID CONTROLLER
            // -------------------------

            float rollCorr  = pidRoll.update(rollRateTarget, imu.getRollRate(), dt);
            float pitchCorr = pidPitch.update(pitchRateTarget, imu.getPitchRate(), dt);

            float yawRateTarget = (yawInput - 1500) * 0.2f;
            float yawCorr = pidYaw.update(yawRateTarget, imu.getYawRate(), dt);


            rollCorr  = constrain(rollCorr,  -150, 150);
            pitchCorr = constrain(pitchCorr, -150, 150);
            yawCorr   = constrain(yawCorr,   -150, 150);


            // MIXER
            mixer.update(rc, rollCorr, pitchCorr, yawCorr);

            // MOTORS
            motor.update(mixer);

            break;
        }


        case FAILSAFE:

            digitalWrite(STATUS_LED_PIN, LOW);
            motor.stopAll();

            if (rc.isValid()) {

                pidRoll.reset();
                pidPitch.reset();
                pidYaw.reset();

                if (wasArmed) {
                    requireArmLow = true;
                    Debug::logln("[STATE] RECOVERED → DISARMED (lower ARM switch)");
                } else {
                    requireArmLow = false;
                    Debug::logln("[STATE] RECOVERED → DISARMED");
                }

                wasArmed = false;
                state = DISARMED;
            }

            return;
    }
}

bool FlightController::canArm() {
    return rc.get(THROTTLE_CH) <= MOTOR_MIN + 100;
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