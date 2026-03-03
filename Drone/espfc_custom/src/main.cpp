#include "Config.h"
#include "WifiManager.h"
#include "MqttManager.h"
#include "RcInput.h"
#include "Mixer.h"
#include "Motor.h"
#include "Debug.h"
#include "IMU.h"
#include "PID.h"

// Global objects
WifiManager wifi;
MqttManager mqtt;
RcInput rc;
Mixer mixer;
Motor motors;
IMU imu;

// PID controllers (values inspired by Betaflight dump)
PID pidRoll(42, 85, 24);   // kp, ki, kd
PID pidPitch(46, 90, 26);
PID pidYaw(45, 90, 0);

unsigned long lastUpdate = 0;

void setup() {
  Serial.begin(115200);
  Debug::logln("[BOOT] ESP-FC Custom Firmware");

  // Network setup
  wifi.begin(WIFI_SSID, WIFI_PASS);
  mqtt.begin(MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, &rc);

  // Hardware setup
  motors.begin();
  imu.begin();   // initialize MPU6050
}

void loop() {
  wifi.update();
  mqtt.update();

  unsigned long now = millis();
  float dt = (now - lastUpdate) / 1000.0f;
  lastUpdate = now;

  imu.update();   // read gyro + accel

  if (rc.isValid()) {
    // Desired rates from RC input
    float rollSet  = rc.get(ROLL_CH);
    float pitchSet = rc.get(PITCH_CH);
    float yawSet   = rc.get(YAW_CH);

    // Measured rates from IMU gyro
    float rollRate  = imu.getGyroX();
    float pitchRate = imu.getGyroY();
    float yawRate   = imu.getGyroZ();

    // PID corrections
    float rollOut  = pidRoll.update(rollSet, rollRate, dt);
    float pitchOut = pidPitch.update(pitchSet, pitchRate, dt);
    float yawOut   = pidYaw.update(yawSet, yawRate, dt);

    // Feed corrections into mixer
    mixer.update(rc, rollOut, pitchOut, yawOut);

    // Drive motors
    motors.update(mixer);
  } else {
    motors.failsafe();
  }

  Debug::heartbeat();
}
