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
    int channels[CHANNELS];
    bool valid = false;
};
