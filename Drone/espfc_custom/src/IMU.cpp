#include "IMU.h"

IMU::IMU() : gyroX(0), gyroY(0), gyroZ(0), accX(0), accY(0), accZ(0) {}

void IMU::begin(int sda, int scl) {
    Wire.begin(sda, scl);
    mpu.initialize();

    if (!mpu.testConnection()) {
        Serial.println("[IMU] MPU6050 connection failed!");
    } else {
        Serial.println("[IMU] MPU6050 ready.");
    }
}

void IMU::update() {
    int16_t ax, ay, az;
    int16_t gx, gy, gz;

    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    // Convert raw gyro to deg/s
    gyroX = gx / 131.0;
    gyroY = gy / 131.0;
    gyroZ = gz / 131.0;

    // Convert raw accel to g
    accX = ax / 16384.0;
    accY = ay / 16384.0;
    accZ = az / 16384.0;
}
