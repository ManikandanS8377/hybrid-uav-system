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

  #if DEBUG_SERIAL
    Serial.begin(115200);
  #endif

  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, LOW);

  delay(2000);
  
  wifi.begin(WIFI_SSID, WIFI_PASS);
  fc.begin();
  Debug::logln("[BOOT] Flight loop initialized");
}

void loop() {
  wifi.update();
  rc.updateConnection();

  unsigned long now = millis();
  if (now - lastLoopMicros < LOOP_PERIOD) return;

  float dt = (now - lastLoopMicros) / 1000.0f;
  lastLoopMicros = now;

  fc.update(dt);
}