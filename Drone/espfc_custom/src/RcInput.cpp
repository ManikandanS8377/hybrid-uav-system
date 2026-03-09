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
    if (WiFi.status() != WL_CONNECTED) return;

    if (!mqttClient.connected()) {
        // Non-blocking reconnect: only attempt once every 2 seconds.
        // mqttClient.connect() is a BLOCKING TCP call that can take 500ms–3000ms
        // on a public broker. Without this cooldown, every dropped connection
        // stalls the entire flight loop during the reconnect attempt, causing
        // lastRcUpdate to go stale → false failsafe immediately after reconnect.
        unsigned long now = millis();
        if (now - lastReconnectAttempt < 2000) {
            return; // cooldown not expired, skip this cycle
        }
        lastReconnectAttempt = now;

        Debug::logln("[MQTT] Attempting connect...");
        digitalWrite(STATUS_LED_PIN, HIGH);

        String clientId = "esp32_" + String(random(0xffff), HEX);
        if (mqttClient.connect(clientId.c_str())) {
            mqttClient.subscribe(MQTT_TOPIC);
            digitalWrite(STATUS_LED_PIN, LOW);
            Debug::logln("[MQTT] Connected + subscribed");

            // Reset timeout clock to now so the 2000ms window starts fresh.
            // The sender needs a moment after broker connect to deliver first packet.
            lastRcUpdate = millis();
        } else {
            Debug::log("[MQTT] Connect failed rc=");
            Debug::logln(mqttClient.state());
            digitalWrite(STATUS_LED_PIN, LOW);
        }
        return; // skip mqttClient.loop() this cycle — just reconnected
    }

    mqttClient.loop(); // process incoming packets
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
    // 2000ms timeout for in-flight use.
    // At 40Hz sender rate, 2000ms = 80 missed packets = definite link loss.
    // Public broker spikes are typically 200–800ms, well within this window.
    //
    // STARTUP GRACE: lastRcUpdate is set to millis() at boot AND reset after
    // each successful MQTT reconnect, so the WiFi+MQTT connect sequence
    // (which can take 3–5 seconds) doesn't consume the timeout window.
    const unsigned long TIMEOUT_MS = 2000;
    return (millis() - lastRcUpdate) <= TIMEOUT_MS;
}

int RcInput::get(int ch) {
    if (ch < 0 || ch >= CHANNELS) return 1500; // safe default
    return channels[ch];
}
