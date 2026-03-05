#include "Debug.h"

void Debug::log(const char* msg) {
  Serial.print(msg);
}

void Debug::logln(const char* msg) {
  Serial.println(msg);
}

void Debug::logln(int val) { 
  Serial.println(val); 
}

void Debug::log(int val) {
  Serial.print(val);
}

void Debug::heartbeat() {
  static unsigned long last = 0;
  unsigned long now = millis();
  if (now - last > 1000) {   // every 1 second
    Serial.println("[HEARTBEAT] Loop alive");
    last = now;
  }
}

void Debug::begin(long baud) { Serial.begin(baud); }

