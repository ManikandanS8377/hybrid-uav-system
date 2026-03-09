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
// FIX #3: Removed duplicate PID objects (pidRoll, pidPitch, pidYaw) that were
// declared here but NEVER used. FlightController owns and runs its own PIDs.
// Having both caused confusion about which gains were actually active.
IMU imu;
Mixer mixer;
Motor motor;
RcInput rc;
FlightController fc;
WifiManager wifi;

// --- Timing ---
unsigned long lastLoopMillis = 0; // renamed from lastLoopMicros (uses millis, not micros)
unsigned long LOOP_PERIOD = 15;   // 15ms = ~66Hz loop rate

// --- Failsafe ---
unsigned long lastRcUpdate = 0;
const unsigned long FAILSAFE_TIMEOUT = 500; // FIX #6: was 5000 (comment said 500, value was wrong)

void setup() {

  #if DEBUG_SERIAL
    Serial.begin(115200);
  #endif

  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, LOW);

  delay(2000);

  // Initialize lastRcUpdate to NOW so the startup gap while WiFi+MQTT
  // is connecting doesn't immediately expire the 2000ms failsafe window.
  // This is reset again after MQTT connects in updateConnection().
  lastRcUpdate = millis();

  wifi.begin(WIFI_SSID, WIFI_PASS);
  fc.begin();
  Debug::logln("[BOOT] Flight loop initialized");
}

void loop() {
  wifi.update();
  // NOTE: rc.updateConnection() is called inside fc.update() → runFlightLoop()
  // Calling it here too was causing mqttClient.loop() to run twice per cycle,
  // which caused message delivery issues and unreliable lastRcUpdate timestamps.

  unsigned long now = millis();
  if (now - lastLoopMillis < LOOP_PERIOD) return;

  float dt = (now - lastLoopMillis) / 1000.0f;
  lastLoopMillis = now;

  fc.update(dt);
}