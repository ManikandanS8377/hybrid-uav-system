#include "WifiManager.h"
#include "Debug.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include "Config.h"

void WifiManager::begin(const char* ssid, const char* pass) {
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, pass);
    Debug::log("[WiFi] Connecting to ");
    Debug::logln(ssid);

    // Wait until connected (blocking only at startup)
    while (WiFi.status() != WL_CONNECTED) {
        digitalWrite(STATUS_LED_PIN, HIGH);
        delay(500);
        Debug::log(".");
        digitalWrite(STATUS_LED_PIN, LOW);
    }
    Debug::logln("\n[WiFi] Connected!");
}

void WifiManager::update() {
  // If already connected, nothing to do
  if (WiFi.status() == WL_CONNECTED) return;

  // Retry logic
  static unsigned long lastAttempt = 0;
  unsigned long now = millis();

  // Try every 2 seconds until connected
  if (now - lastAttempt > 2000) {
    Debug::logln("[WiFi] Not connected, retrying...");
    WiFi.reconnect();   // triggers another attempt
    lastAttempt = now;
  }
}
