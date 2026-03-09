#include "FlightController.h"
#include "Debug.h"

extern unsigned long LOOP_PERIOD;

uint32_t armStartTime = 0;

void FlightController::begin() {
    imu.begin();
    motor.begin();
    rc.begin();

    // FIX #2: Reduced PID gains from 1.0 to safe starting values
    // kp=1.0 was too aggressive on gyro rate, causing ±300+ corrections → flip
    pidRoll.init(0.3, 0.0, 0.002);
    pidPitch.init(0.3, 0.0, 0.002);
    pidYaw.init(0.5, 0.0, 0.0);

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

    int armSwitch = rc.get(ARM_CH);

    // ARM thresholds use clear boundaries away from CENTER (1500):
    //   > 1700 = definitively HIGH (armed)
    //   < 1300 = definitively LOW  (disarmed)
    //   1300–1700 = neutral dead zone — ignored, no state change
    // This prevents the 1500 dead-zone bug where armSwitch == CENTER
    // could neither arm nor disarm the drone.
    bool armHigh = (armSwitch > 1700);
    bool armLow  = (armSwitch < 1300);

    switch (state) {
        case DISARMED:
            // requireArmLow: after failsafe recovery, ARM switch must return to LOW
            // before a new arm is allowed. Prevents instant re-arm loop when
            // FAILSAFE recovers while ARM is still held high.
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
            // FIX: idle spin during arming avoids jerk from 0→throttle on ARMED transition
            motor.idle();
            if (millis() - armStartTime > 1000) {
                // FIX: reset PID integrals before every arm to clear any windup from last session
                pidRoll.reset();
                pidPitch.reset();
                pidYaw.reset();
                wasArmed = true;   // mark that motors have actually run
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

            // --- Get raw stick values ---
            int rollInput  = rc.get(ROLL_CH);
            int pitchInput = rc.get(PITCH_CH);
            int yawInput   = rc.get(YAW_CH);

            // --- Apply deadband (±20 around center 1500) ---
            // CENTER aligned to 1500 matching mqtt_sender.py
            if (abs(rollInput  - 1500) < 20) rollInput  = 1500;
            if (abs(pitchInput - 1500) < 20) pitchInput = 1500;
            if (abs(yawInput   - 1500) < 20) yawInput   = 1500;

            // --- Convert to rate setpoints ---
            float rollSetpoint  = (rollInput  - 1500) * 0.2f;
            float pitchSetpoint = (pitchInput - 1500) * 0.2f;
            float yawSetpoint   = (yawInput   - 1500) * 0.2f;

            // PID
            float rollCorr  = pidRoll.update(rollSetpoint,  imu.getRollRate(), dt);
            float pitchCorr = pidPitch.update(pitchSetpoint, imu.getPitchRate(), dt);
            float yawCorr   = pidYaw.update(yawSetpoint,    imu.getYawRate(), dt);

            // FIX #4: Tighter clamp ±150 (was ±300, too wide for mini brushed drone)
            rollCorr  = constrain(rollCorr,  -150, 150);
            pitchCorr = constrain(pitchCorr, -150, 150);
            yawCorr   = constrain(yawCorr,   -150, 150);

            // MIXER
            mixer.update(rc, rollCorr, pitchCorr, yawCorr);

            // send to motors
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
                // Only require ARM to go low if we were actually ARMED when
                // failsafe triggered. If failsafe fired during DISARMED/ARMING
                // (e.g. startup gap before first MQTT packet), don't block re-arm —
                // the user never had motors running so there's no safety risk.
                if (wasArmed) {
                    requireArmLow = true;
                    Debug::logln("[STATE] RECOVERED → DISARMED (lower ARM switch to re-arm)");
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
