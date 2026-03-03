#pragma once
#include <Arduino.h>
#include <Wire.h>
#include <MPU6050.h>

class IMU {
public:
    IMU();
    void begin(int sda = 21, int scl = 22);   // default ESP32 I2C pins
    void update();                            // read latest sensor values

    // Accessors
    float getGyroX() const { return gyroX; }
    float getGyroY() const { return gyroY; }
    float getGyroZ() const { return gyroZ; }
    float getAccX() const { return accX; }
    float getAccY() const { return accY; }
    float getAccZ() const { return accZ; }

private:
    MPU6050 mpu;
    float gyroX, gyroY, gyroZ;
    float accX, accY, accZ;
};
