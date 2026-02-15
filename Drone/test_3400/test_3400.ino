#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

// --- Motor Pin Mapping ---
#define M1 25 // Front Right (Prop B - CCW)
#define M2 26 // Front Left  (Prop A - CW)
#define M3 27 // Back Left   (Prop B - CCW)
#define M4 14 // Back Right  (Prop A - CW)

const int pwmFreq = 16000;
const int pwmRes  = 8;

// --- PID TUNING ---
float Kp = 0.8; 
float Kd = 0.05; 

// --- YOUR MEASURED GROUND VALUES ---
float targetPitch = 4.0; 
float targetRoll  = 0.6;

float roll, pitch, rollPrev, pitchPrev;
unsigned long prevTime;
unsigned long missionStart;
bool armed = false;

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  mpu.initialize();

  ledcAttach(M1, pwmFreq, pwmRes);
  ledcAttach(M2, pwmFreq, pwmRes);
  ledcAttach(M3, pwmFreq, pwmRes);
  ledcAttach(M4, pwmFreq, pwmRes);
  
  // Safety: Kill motors immediately on boot
  stopMotors();

  Serial.println("--- READY FOR 10-FOOT MISSION ---");
  Serial.println("Place drone on flat ground. Launching in 5s...");
  delay(5000);

  missionStart = millis();
  prevTime = millis();
  armed = true;
}

void loop() {
  if (!armed) return;

  unsigned long currentTime = millis();
  float dt = (currentTime - prevTime) / 1000.0;
  if (dt <= 0 || dt > 0.1) dt = 0.01;
  prevTime = currentTime;

  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // 1. Calculate Current Angles (Filtered)
  float accRoll  = (atan2(ay, az) * 57.3) * -1.0;
  float accPitch = atan2(-ax, az) * 57.3;

  roll  = 0.96 * (roll - (gx / 131.0) * dt) + 0.04 * accRoll;
  pitch = 0.96 * (pitch + (gy / 131.0) * dt) + 0.04 * accPitch;

  // 2. Mission Logic (Thrust Control)
  unsigned long activeTime = millis() - missionStart;
  int baseSpeed = 0;

  if (activeTime < 3000) {
    baseSpeed = 50; 
  } else if (activeTime < 15000) {
    baseSpeed = 230; // Power for climb phase
  } else {
    baseSpeed = 0;   
    armed = false;
    stopMotors();
    Serial.println("MISSION COMPLETE - SHUTTING DOWN");
  }

  // 3. PID Math (Error = Target - Current)
  float pError = targetPitch - pitch; 
  float rError = targetRoll - roll;

  float pPID = (Kp * pError) + (Kd * (pError - pitchPrev) / dt);
  float rPID = (Kp * rError) + (Kd * (rError - rollPrev) / dt);

  pitchPrev = pError;
  rollPrev = rError;

  // 4. Motor Mixing
  // Based on your values: If pitch > 4.0 (Front Down), pError becomes negative.
  // Using (baseSpeed - pPID) makes the front motors go FASTER.
  int s1 = constrain(baseSpeed - pPID + rPID, 0, 255); // Front Right
  int s2 = constrain(baseSpeed - pPID - rPID, 0, 255); // Front Left
  int s3 = constrain(baseSpeed + pPID - rPID, 0, 255); // Back Left
  int s4 = constrain(baseSpeed + pPID + rPID, 0, 255); // Back Right

  if (baseSpeed > 0) {
    ledcWrite(M1, s1); ledcWrite(M2, s2); ledcWrite(M3, s3); ledcWrite(M4, s4);
  }
}

void stopMotors() {
  ledcWrite(M1, 0); ledcWrite(M2, 0); ledcWrite(M3, 0); ledcWrite(M4, 0);
}