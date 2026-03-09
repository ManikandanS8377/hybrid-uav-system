#include "IMU.h"
#include "Debug.h"

IMU::IMU() 
    : gyroX(0), gyroY(0), gyroZ(0),
      accX(0), accY(0), accZ(0),
      angleX(0), angleY(0),
      gyroOffsetX(0), gyroOffsetY(0), gyroOffsetZ(0),
      angleOffsetX(0), angleOffsetY(0) {}

void IMU::begin(int sda, int scl) {
    Wire.begin(sda, scl);
    mpu.initialize();

    if (!mpu.testConnection()) {
        Debug::logln("[IMU] MPU6050 connection failed!");
    }

    Debug::logln("[IMU] Calibrating gyro... Keep still!");

    long sumX = 0, sumY = 0, sumZ = 0;
    const int samples = 1000;

    for (int i = 0; i < samples; i++) {
        int16_t ax, ay, az;
        int16_t gx, gy, gz;

        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

        sumX += gx;
        sumY += gy;
        sumZ += gz;

        delay(2);
    }

    gyroOffsetX = (sumX / (float)samples) / 131.0f;
    gyroOffsetY = (sumY / (float)samples) / 131.0f;
    gyroOffsetZ = (sumZ / (float)samples) / 131.0f;

    Debug::logln("[IMU] Gyro calibration done");

    // FIX #5: Run filter for 500 cycles so angles actually settle before capturing offset
    // Old: delay(200) captured angleX/Y = 0 (filter hadn't run), making offset useless
    Debug::logln("[IMU] Settling filter...");
    for (int i = 0; i < 500; i++) {
        update(0.002f);
        delay(2);
    }

    // Capture true resting angle as zero reference
    angleOffsetX = angleX;
    angleOffsetY = angleY;
    Debug::logln("[IMU] Angle offset captured");
}

void IMU::update(float dt) {
    int16_t ax, ay, az;
    int16_t gx, gy, gz;

    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    // Convert raw gyro to deg/s
    gyroX = (gx / 131.0f) - gyroOffsetX;
    gyroY = (gy / 131.0f) - gyroOffsetY;
    gyroZ = (gz / 131.0f) - gyroOffsetZ;

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
