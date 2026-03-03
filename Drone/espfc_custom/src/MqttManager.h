#pragma once
#include <Arduino.h>
#include "RcInput.h"

class MqttManager {
public:
  void begin(const char* broker, int port, const char* topic, RcInput* rc);
  void update();

private:
  static void callback(char* topic, byte* payload, unsigned int length);
  const char* _topic;
};
