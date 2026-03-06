#include "Debug.h"
#include "Config.h"


void Debug::log(const char* msg) {
#if DEBUG_SERIAL
  Serial.print(msg);
#endif
}

void Debug::logln(const char* msg) {
#if DEBUG_SERIAL
  Serial.println(msg);
#endif
}

void Debug::logln(int val) {
#if DEBUG_SERIAL
  Serial.println(val);
#endif
}

void Debug::log(int val) {
#if DEBUG_SERIAL
  Serial.print(val);
#endif
}

void Debug::heartbeat() {
#if DEBUG_SERIAL
  static unsigned long last = 0;
  unsigned long now = millis();
  if (now - last > 1000) {
    Serial.println("[HEARTBEAT] Loop alive");
    last = now;
  }
#endif
}

void Debug::begin(long baud) {
#if DEBUG_SERIAL
  Serial.begin(baud);
#endif
}