#include "IMU.h"

IMU::IMU() 
    : gyroX(0), gyroY(0), gyroZ(0),
      accX(0), accY(0), accZ(0),
      angleX(0), angleY(0) {}

void IMU::begin(int sda, int scl) {
    Wire.begin(sda, scl);
    mpu.initialize();

    if (!mpu.testConnection()) {
        Serial.println("[IMU] MPU6050 connection failed!");
    } else {
        Serial.println("[IMU] MPU6050 ready.");
    }
}

void IMU::update(float dt) {
    int16_t ax, ay, az;
    int16_t gx, gy, gz;

    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    // Convert raw gyro to deg/s
    gyroX = gx / 131.0f;
    gyroY = gy / 131.0f;
    gyroZ = gz / 131.0f;

    // Convert raw accel to g
    accX = ax / 16384.0f;
    accY = ay / 16384.0f;
    accZ = az / 16384.0f;

    // Calculate accelerometer angles (in degrees)
    float accAngleX = atan2(accY, accZ) * RAD_TO_DEG;
    float accAngleY = atan2(-accX, accZ) * RAD_TO_DEG;

    // Complementary filter: fuse gyro + accel
    angleX = 0.98f * (angleX + gyroX * dt) + 0.02f * accAngleX;
    angleY = 0.98f * (angleY + gyroY * dt) + 0.02f * accAngleY;
}
