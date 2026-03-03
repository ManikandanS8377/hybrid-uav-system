#include "Mixer.h"
#include "Config.h"
#include "RcInput.h"
#include "Debug.h"

void Mixer::update(RcInput& rc) {
  int throttle = rc.get(THROTTLE_CH);
  int roll     = rc.get(ROLL_CH) - 1500;
  int pitch    = rc.get(PITCH_CH) - 1500;
  int yaw      = rc.get(YAW_CH) - 1500;
  bool armed   = rc.get(ARM_CH) > 1500;

  if (!armed || throttle < MOTOR_IDLE) {
    for (int i = 0; i < 4; i++) outputs[i] = MOTOR_MIN;
    return;
  }

  // QUADX mix (raw RC only)
  outputs[0] = throttle + pitch + roll - yaw; // front right
  outputs[1] = throttle + pitch - roll + yaw; // front left
  outputs[2] = throttle - pitch + roll + yaw; // rear right
  outputs[3] = throttle - pitch - roll - yaw; // rear left

  for (int i = 0; i < 4; i++) {
    if (outputs[i] < MOTOR_MIN) outputs[i] = MOTOR_MIN;
    if (outputs[i] > MOTOR_MAX) outputs[i] = MOTOR_MAX;
  }
}

// NEW overload: include PID corrections
void Mixer::update(RcInput& rc, float rollCorr, float pitchCorr, float yawCorr) {
  int throttle = rc.get(THROTTLE_CH);
  bool armed   = rc.get(ARM_CH) > 1500;

  if (!armed || throttle < MOTOR_IDLE) {
    for (int i = 0; i < 4; i++) outputs[i] = MOTOR_MIN;
    return;
  }

  // QUADX mix (RC throttle + PID corrections)
  outputs[0] = throttle + pitchCorr + rollCorr - yawCorr; // front right
  outputs[1] = throttle + pitchCorr - rollCorr + yawCorr; // front left
  outputs[2] = throttle - pitchCorr + rollCorr + yawCorr; // rear right
  outputs[3] = throttle - pitchCorr - rollCorr - yawCorr; // rear left

  for (int i = 0; i < 4; i++) {
    if (outputs[i] < MOTOR_MIN) outputs[i] = MOTOR_MIN;
    if (outputs[i] > MOTOR_MAX) outputs[i] = MOTOR_MAX;
  }

  // Debug print
  for (int i = 0; i < 4; i++) {
    Debug::log("[Mixer] Motor "); Debug::log(i);
    Debug::log("="); Debug::logln(outputs[i]);
  }
}
