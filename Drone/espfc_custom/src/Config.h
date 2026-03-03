#pragma once

// WiFi + MQTT
#define WIFI_SSID     "Manikandan"
#define WIFI_PASS     "M@nik@nd@n"
#define MQTT_BROKER   "broker.emqx.io"
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

// Motor pins (adjust to your ESP32 GPIOs)
const int MOTOR_PINS[4] = {14, 25, 27, 26}; // AO3400 gate pins
