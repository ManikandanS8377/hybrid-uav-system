#pragma once
#include <Arduino.h>
#include "IMU.h"
#include "RcInput.h"
#include "Motor.h"
#include "PID.h"
#include "Mixer.h"

enum FlightState {
    DISARMED,
    ARMING,
    ARMED,
    FAILSAFE
};

class FlightController {
public:
    void begin();
    void update(float dt);              // main flight loop
    void setFailsafe();         // trigger failsafe
    void runFlightLoop(float dt);       // executes one control cycle
    FlightState getState() const { return state; }

private:
    bool canArm();              // throttle + conditions check
    void arm();
    void disarm();

    FlightState state = DISARMED;
    uint32_t lastLoop = 0;
    const uint32_t LOOP_US = 2000; // 500 Hz

    // Modules
    IMU imu;
    RcInput rc;
    Motor motor;
    PID pidRoll, pidPitch, pidYaw;
    Mixer mixer;

    // Safety
    unsigned long lastRcPacket = 0;
    const unsigned long FAILSAFE_TIMEOUT = 1000; // ms
};
