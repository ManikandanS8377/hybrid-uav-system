#pragma once
#include "Config.h"
#include <Arduino.h>

class RcInput {
public:
    void begin();                     // configure WiFi/MQTT
    void updateConnection();           // non-blocking reconnect + loop
    void process(const char* payload); // parse incoming RC frame
    bool isValid();                    // check if last frame was valid
    int get(int ch);                   // get channel value (1000–2000)

private:
    // Initialize all channels to safe defaults.
    // C++ does NOT zero-initialize plain int arrays — garbage values on first read
    // caused armLow = (channels[4] < 1300) to be true on the very first ARMED check,
    // triggering an immediate disarm before any packet was ever processed.
    int channels[CHANNELS] = {1500, 1500, 1000, 1500, 1000, 1500, 1500, 1500};
    //                         ROLL  PITCH  THR  YAW   ARM  AUX2  AUX3  AUX4
    // THR=1000 and ARM=1000 are the safe "disarmed, throttle low" defaults.
    bool valid = false;
    unsigned long lastReconnectAttempt = 0; // non-blocking reconnect cooldown
};
