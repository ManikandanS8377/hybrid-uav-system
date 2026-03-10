#pragma once

// WiFi + MQTT
#define WIFI_SSID     "Manikandan"
#define WIFI_PASS     "M@nik@nd@n"

// ── BROKER SELECTION ──────────────────────────────────────────
// PUBLIC broker (broker.emqx.io) loses ~80-95% of packets over internet.
// This causes FAILSAFE during flight because gaps between received packets
// exceed the 2000ms timeout window — even though sender is sending at 40Hz.
//
// USE LOCAL BROKER INSTEAD:
// 1. Run tools/setup_local_broker.bat on your Windows PC
// 2. Find your PC's WiFi IP (shown by the script, e.g. 192.168.1.x)
// 3. Replace "broker.emqx.io" below with your PC's IP address
// 4. Make sure ESP32 and PC are on the same WiFi network (Manikandan)
// ─────────────────────────────────────────────────────────────
#define MQTT_BROKER   "192.168.43.222"   // ← CHANGE to your PC's IP, e.g. "192.168.1.100"
#define MQTT_PORT     1883
#define MQTT_TOPIC    "espfc/rc"

// RC channels
const int CHANNELS = 8;
#define THROTTLE_CH   2
#define ROLL_CH       0
#define PITCH_CH      1
#define YAW_CH        3
#define ARM_CH        4

// Motor settings
#define MOTOR_MIN     1000
#define MOTOR_IDLE    1070
#define MOTOR_MAX     2000

// LED Indicator pins
#define STATUS_LED_PIN 2
#define DEBUG_SERIAL 1   // TEMP: enabled to verify motor mixer corrections

// Motor pins (adjust to your ESP32 GPIOs)
const int MOTOR_PINS[4] = {14, 25, 27, 26}; // AO3400 gate pins
