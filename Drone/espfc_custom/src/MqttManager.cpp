#include "MqttManager.h"
#include "Debug.h"
#include <WiFi.h>
#include <PubSubClient.h>

WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);   // ✅ match RcInput.cpp extern
RcInput* rcInput;

void MqttManager::begin(const char* broker, int port, const char* topic, RcInput* rc) {
  mqttClient.setServer(broker, port);
  mqttClient.setCallback(callback);
  rcInput = rc;
  _topic = topic;
}

void MqttManager::update() {
  if (!mqttClient.connected()) {
    if (WiFi.status() == WL_CONNECTED) {
      Debug::logln("[MQTT] Connecting...");
      if (mqttClient.connect("espfc")) {
        Debug::logln("[MQTT] Connected");
        mqttClient.subscribe(_topic);
      }
    }
  }
  mqttClient.loop();
}

void MqttManager::callback(char* topic, byte* payload, unsigned int length) {
  if (!rcInput) return;
  char buf[128];
  memcpy(buf, payload, length);
  buf[length] = 0;
  rcInput->process(buf);
}
