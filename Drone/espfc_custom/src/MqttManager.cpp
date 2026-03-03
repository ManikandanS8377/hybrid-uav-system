#include "MqttManager.h"
#include "Debug.h"
#include <WiFi.h>
#include <PubSubClient.h>

WiFiClient wifiClient;
PubSubClient client(wifiClient);
RcInput* rcInput;

void MqttManager::begin(const char* broker, int port, const char* topic, RcInput* rc) {
  client.setServer(broker, port);
  client.setCallback(callback);
  rcInput = rc;
  _topic = topic;
}

void MqttManager::update() {
  if (!client.connected()) {
    if (WiFi.status() == WL_CONNECTED) {
      Debug::logln("[MQTT] Connecting...");
      if (client.connect("espfc")) {
        Debug::logln("[MQTT] Connected");
        client.subscribe(_topic);
      }
    }
  }
  client.loop();
}

void MqttManager::callback(char* topic, byte* payload, unsigned int length) {
  if (!rcInput) return;
  char buf[128];
  memcpy(buf, payload, length);
  buf[length] = 0;
  rcInput->process(buf);
}
