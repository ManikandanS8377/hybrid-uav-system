#pragma once
#include <Arduino.h>

class Debug {
public:
  static void log(const char* msg);
  static void logln(const char* msg);
  static void logln(int val);
  static void log(int val);
  static void heartbeat();
  static void begin(long baud);

};
