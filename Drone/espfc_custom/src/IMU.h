#pragma once
#include <Arduino.h>
#include <Wire.h>
#include <MPU6050.h>

class IMU {
public:
    IMU();
    void begin(int sda = 21, int scl = 22); // default ESP32 I2C pins
    void update(float dt);

    // Accessors
    float getRollRate() const { return gyroX; }
    float getPitchRate() const { return gyroY; }
    float getYawRate() const { return gyroZ; }
    float getRollAngle() const { return angleX; }
    float getPitchAngle() const { return angleY; }

private:
    MPU6050 mpu;

    // Raw sensor values
    float gyroX, gyroY, gyroZ;
    float accX, accY, accZ;

    // Filtered angles
    float angleX, angleY;
};
