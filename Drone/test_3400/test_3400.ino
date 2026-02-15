#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

// ===== MOTOR PINS (Corrected Mapping) =====
#define M1 25  // Front Right
#define M2 26  // Front Left
#define M3 14  // Back Left   (SWAPPED)
#define M4 27  // Back Right  (SWAPPED)

// ===== PID TUNING =====
float Kp = 1.4;
float Ki = 0.02;
float Kd = 0.05;

// ===== IMU VARIABLES =====
float roll = 0, pitch = 0;
float rollOffset = 0, pitchOffset = 0;

float pI = 0, rI = 0;
float prevPitchError = 0, prevRollError = 0;

unsigned long prevTime;
unsigned long missionStart;

bool initialized = false;
bool armed = false;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  mpu.initialize();

  // Attach PWM
  ledcAttach(M1, 16000, 8);
  ledcAttach(M2, 16000, 8);
  ledcAttach(M3, 16000, 8);
  ledcAttach(M4, 16000, 8);

  stopMotors();

  Serial.println("Keep drone perfectly flat...");
  delay(4000);

  // ===== INITIAL ANGLE CALIBRATION =====
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  roll  = atan2(ay, az) * 57.2958;
  pitch = atan2(-ax, sqrt(ay * ay + az * az)) * 57.2958;

  rollOffset  = roll;
  pitchOffset = pitch;

  prevTime = millis();
  missionStart = millis();
  armed = true;

  Serial.println("MISSION STARTED");
}

void loop() {

  if (!armed) return;

  unsigned long now = millis();
  float dt = (now - prevTime) / 1000.0;
  prevTime = now;

  if (dt <= 0 || dt > 0.1) dt = 0.01;

  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // ===== ANGLE CALCULATION =====
  float accRoll  = atan2(ay, az) * 57.2958;
  float accPitch = atan2(-ax, sqrt(ay * ay + az * az)) * 57.2958;

  float gyroRollRate  = gx / 131.0;
  float gyroPitchRate = gy / 131.0;

  // Complementary filter
  roll  = 0.98 * (roll + gyroRollRate * dt) + 0.02 * accRoll;
  pitch = 0.98 * (pitch + gyroPitchRate * dt) + 0.02 * accPitch;

  float correctedRoll  = roll  - rollOffset;
  float correctedPitch = pitch - pitchOffset;

  // ===== MISSION LOGIC =====
  unsigned long t = millis() - missionStart;
  int baseSpeed = 0;

  if (t < 3000) {
    baseSpeed = 50;   // Idle spin
  }
  else if (t < 13000) {
    baseSpeed = 230;  // Climb
  }
  else {
    baseSpeed = 200;  // Hover (adjust after test)
  }

  // ===== PID =====
  float pitchError = 0 - correctedPitch;
  float rollError  = 0 - correctedRoll;

  pI += pitchError * dt;
  rI += rollError * dt;

  pI = constrain(pI, -20, 20);
  rI = constrain(rI, -20, 20);

  float pPID = Kp * pitchError +
               Ki * pI +
               Kd * (pitchError - prevPitchError) / dt;

  float rPID = Kp * rollError +
               Ki * rI +
               Kd * (rollError - prevRollError) / dt;

  prevPitchError = pitchError;
  prevRollError  = rollError;

  // ===== MOTOR MIXING =====
  int s1 = constrain(baseSpeed - pPID + rPID, 0, 255); // Front Right
  int s2 = constrain(baseSpeed - pPID - rPID, 0, 255); // Front Left
  int s3 = constrain(baseSpeed + pPID - rPID, 0, 255); // Back Left
  int s4 = constrain(baseSpeed + pPID + rPID, 0, 255); // Back Right

  ledcWrite(M1, s1);
  ledcWrite(M2, s2);
  ledcWrite(M3, s3);
  ledcWrite(M4, s4);
}

void stopMotors() {
  ledcWrite(M1, 0);
  ledcWrite(M2, 0);
  ledcWrite(M3, 0);
  ledcWrite(M4, 0);
}