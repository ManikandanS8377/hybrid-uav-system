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
unsigned long LOOP_PERIOD = 15; // 

// --- Failsafe ---
unsigned long lastRcUpdate = 0;
const unsigned long FAILSAFE_TIMEOUT = 5000; // 500 ms

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

  unsigned long now = millis();

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

  // --- Flight control loop ---
  imu.update(dt);   
  fc.update(dt);

  // PID corrections (gyro rates in deg/s)
  float rollCorr  = pidRoll.update(0, imu.getRollRate(), dt);
  float pitchCorr = pidPitch.update(0, imu.getPitchRate(), dt);
  float yawCorr   = pidYaw.update(0, imu.getYawRate(), dt);

  // Mixer
  mixer.update(rc, rollCorr, pitchCorr, yawCorr);

  // Motors
  motor.update(mixer);
}
