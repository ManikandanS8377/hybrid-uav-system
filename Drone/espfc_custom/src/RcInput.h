
#include "Config.h"
#pragma once
#include <Arduino.h>

class RcInput {
public:
  void process(const char* payload);
  bool isValid();
  int get(int ch);

private:
  int channels[CHANNELS];
  bool valid;
};
