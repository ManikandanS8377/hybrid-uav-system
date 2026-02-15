#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

// --- Motor Pin Mapping ---
#define M1 25 // Front Right (Prop B)
#define M2 26 // Front Left  (Prop A)
#define M3 27 // Back Left   (Prop B)
#define M4 14 // Back Right  (Prop A)

const int pwmFreq = 16000;
const int pwmRes  = 8;

// --- PID Tuning ---
float Kp = 1.3; 
float Kd = 0.05; 

// --- Calibration & Timing ---
float roll, pitch, rollPrev, pitchPrev;
float pitchOffset = 0, rollOffset = 0;
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
  
  stopMotors();

  Serial.println("!!! CALIBRATION MODE !!!");
  Serial.println("Hold drone so MOTORS are perfectly level.");
  
  // Average 100 readings to find the 'Ground Offset'
  float sumP = 0, sumR = 0;
  for(int i = 0; i < 100; i++) {
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    sumP += atan2(-ax, az) * 57.3;
    sumR += (atan2(ay, az) * 57.3) * -1.0;
    delay(20);
    if(i % 20 == 0) Serial.print("."); 
  }
  
  pitchOffset = sumP / 100.0;
  rollOffset = sumR / 100.0;
  
  Serial.println("\nCalibration Complete!");
  Serial.print("Pitch Offset: "); Serial.println(pitchOffset);
  Serial.println("ARMING IN 2 SECONDS... STAND BACK!");
  delay(2000);

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

  // 1. Read Sensors
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // 2. Calculate Angles with Calibration Offset
  float accRoll  = (atan2(ay, az) * 57.3) * -1.0;
  float accPitch = atan2(-ax, az) * 57.3;

  // Complementary Filter
  roll  = (0.96 * (roll - (gx / 131.0) * dt) + 0.04 * accRoll) - rollOffset;
  pitch = (0.96 * (pitch + (gy / 131.0) * dt) + 0.04 * accPitch) - pitchOffset;

  // 3. Mission Logic (10 Foot Goal)
  unsigned long activeTime = millis() - missionStart;
  int baseSpeed = 0;

  if (activeTime < 300) {
    baseSpeed = 50; 
  } else if (activeTime < 15000) {
    baseSpeed = 195; // Thrust for 10-foot climb
  } else {
    baseSpeed = 0;   // Shut down
    armed = false;
    stopMotors();
    Serial.println("MISSION END");
  }

  // 4. PID Math
  float pError = 0 - pitch;
  float rError = 0 - roll;

  float pPID = (Kp * pError) + (Kd * (pError - pitchPrev) / dt);
  float rPID = (Kp * rError) + (Kd * (rError - rollPrev) / dt);

  pitchPrev = pError;
  rollPrev = rError;

  // 5. Motor Mixing
  int s1 = constrain(baseSpeed - pPID + rPID, 0, 255); // Front Right
  int s2 = constrain(baseSpeed - pPID - rPID, 0, 255); // Front Left
  int s3 = constrain(baseSpeed + pPID - rPID, 0, 255); // Back Left
  int s4 = constrain(baseSpeed + pPID + rPID, 0, 255); // Back Right

  // 6. Output
  if (baseSpeed > 0) {
    ledcWrite(M1, s1); ledcWrite(M2, s2); ledcWrite(M3, s3); ledcWrite(M4, s4);
  }

  // 7. Debug
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 500) {
    Serial.print("Pitch: "); Serial.print(pitch, 1);
    Serial.print(" | Roll: "); Serial.print(roll, 1);
    Serial.print(" | Motors: "); Serial.print(s1); Serial.print(","); Serial.println(s2);
    lastPrint = millis();
  }
}

void stopMotors() {
  ledcWrite(M1, 0); ledcWrite(M2, 0); ledcWrite(M3, 0); ledcWrite(M4, 0);
}