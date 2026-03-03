#include <Arduino.h>
#include "IMU.h"
#include "PID.h"
#include "Mixer.h"
#include "Motor.h"
#include "RcInput.h"
#include "Debug.h"
#include "FlightController.h"
#include "WifiManager.h"
#include "config.h"


// --- Global objects ---
IMU imu;
PID pidRoll(0.5, 0.0, 0.1);
PID pidPitch(0.5, 0.0, 0.1);
PID pidYaw(0.5, 0.0, 0.1);
Mixer mixer;
Motor motor;
RcInput rc;
FlightController fc;
WifiManager wifi;

// --- Timing ---
unsigned long lastLoopMicros = 0;
const unsigned long LOOP_PERIOD = 2000; // 2000 µs = 500 Hz loop

// --- Failsafe ---
unsigned long lastRcUpdate = 0;
const unsigned long FAILSAFE_TIMEOUT = 500000; // 500 ms

void setup() {
  Serial.begin(115200);
  wifi.begin(WIFI_SSID, WIFI_PASS);
  imu.begin();
  motor.begin();
  rc.begin();
  fc.begin();
  Debug::logln("[BOOT] Flight loop initialized");
}

void loop() {
  // Non-blocking WiFi/MQTT handling
  wifi.update();
  rc.updateConnection();

  unsigned long now = micros();

  // Enforce fixed loop frequency
  if (now - lastLoopMicros < LOOP_PERIOD) return;
  float dt = (now - lastLoopMicros) / 1e6f; // seconds
  lastLoopMicros = now;

  // --- Failsafe check ---
  if ((now - lastRcUpdate) > FAILSAFE_TIMEOUT) {
    motor.failsafe();
    Debug::logln("[FAILSAFE] RC timeout, motors off");
    return;
  }

  fc.runFlightLoop(dt);

  // --- Flight control loop ---
  imu.update(dt);   // ✅ pass dt for complementary filter
  fc.update(dt);

  // RC inputs
  int throttle = rc.get(THROTTLE_CH);
  bool armed   = rc.get(ARM_CH) > 1500;

  if (!armed || throttle < MOTOR_IDLE) {
    motor.idle();
    return;
  }

  // PID corrections (gyro rates in deg/s)
  float rollCorr  = pidRoll.update(0, imu.getRollRate(), dt);
  float pitchCorr = pidPitch.update(0, imu.getPitchRate(), dt);
  float yawCorr   = pidYaw.update(0, imu.getYawRate(), dt);

  // Mixer
  mixer.update(rc, rollCorr, pitchCorr, yawCorr);

  // Motors
  motor.update(mixer);
}
