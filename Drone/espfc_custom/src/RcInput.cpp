#include "RcInput.h"
#include "Debug.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include "config.h"

// Use extern to reference global clients defined elsewhere (e.g. MqttManager.cpp)
extern WiFiClient wifiClient;
extern PubSubClient mqttClient;

// External failsafe timestamp (declared in main.cpp)
extern unsigned long lastRcUpdate;

void RcInput::begin() {
    // Configure MQTT server + callback
    mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
    mqttClient.setCallback([this](char* topic, byte* payload, unsigned int length) {
        char buf[128];
        if (length >= sizeof(buf)) {
            Debug::logln("[RcInput] Payload too long, ignoring");
            return;
        }
        strncpy(buf, (char*)payload, length);
        buf[length] = '\0';

        this->process(buf);
        lastRcUpdate = millis();
    });
}

void RcInput::updateConnection() {
    if (WiFi.status() == WL_CONNECTED) {
        if (!mqttClient.connected()) {
            Debug::logln("[MQTT] Attempting connect...");
            String clientId = "esp32_client_" + String(random(0xffff), HEX);
            if (mqttClient.connect(clientId.c_str())) {
                mqttClient.subscribe(MQTT_TOPIC);
                Debug::logln("[MQTT] Connected + subscribed");
            } else {
                Debug::log("[MQTT] Connect failed, rc=");
                Debug::logln(mqttClient.state());
            }
        }
        mqttClient.loop(); // process packets
    }
}


void RcInput::process(const char* payload) {
    valid = false;

    int idx = 0;
    char buf[128];
    strncpy(buf, payload, sizeof(buf));
    buf[sizeof(buf)-1] = '\0';

    char* token = strtok(buf, ",");
    while (token != nullptr && idx < CHANNELS) {
        channels[idx] = atoi(token);
        idx++;
        token = strtok(nullptr, ",");
    }

    if (idx == CHANNELS) {
        valid = true;
        lastRcUpdate = millis();
        Debug::log("[RcInput] Frame OK: ");
        Debug::logln(payload);
    } else {
        Debug::log("[RcInput] Frame ERR: ");
        Debug::logln(payload);
    }
}

bool RcInput::isValid() {
    const unsigned long TIMEOUT = 1500; // 1500ms
    return (millis() - lastRcUpdate) <= TIMEOUT;
}

int RcInput::get(int ch) {
    if (ch < 0 || ch >= CHANNELS) return 1500; // safe default
    return channels[ch];
}
