#pragma once
#include <Arduino.h>

class WifiManager {
public:
  void begin(const char* ssid, const char* pass);
  void update();
};
