#include "Mixer.h"
#include "Config.h"
#include "Debug.h"

// FIX: Removed internal ARM_CH re-check from Mixer.
// FlightController state machine is the single authority on armed/disarmed.
// Re-reading ARM_CH here caused motors to cut if arm switch glitched for 1 frame,
// even while legitimately in ARMED state.
void Mixer::update(RcInput& rc, float rollCorr, float pitchCorr, float yawCorr)
{
    int throttle = rc.get(THROTTLE_CH);

    // ensure idle spin at minimum when armed (Mixer is only called when ARMED)
    throttle = max(throttle, (int)MOTOR_IDLE);

    // QUADX mix — equations matched to physical wiring and actual spin directions:
    //   index 0 → GPIO 14 → Rear  RIGHT (CW)
    //   index 1 → GPIO 25 → Front RIGHT (CCW)
    //   index 2 → GPIO 27 → Rear  LEFT  (CCW)
    //   index 3 → GPIO 26 → Front LEFT  (CW)
    //
    //         FRONT
    //   FL(CW)   FR(CCW)
    //   RL(CCW)  RR(CW)
    //         REAR
    //
    // Roll  right(+) → right side needs LESS power
    // Pitch fwd  (+) → front  side needs LESS power
    // Yaw   CW   (+) → CW motors (RR,FL) MORE, CCW motors (FR,RL) LESS
    outputs[0] = throttle + rollCorr - pitchCorr + yawCorr; // Rear  RIGHT (CW)
    outputs[1] = throttle + rollCorr + pitchCorr - yawCorr; // Front RIGHT (CCW)
    outputs[2] = throttle - rollCorr - pitchCorr - yawCorr; // Rear  LEFT  (CCW)
    outputs[3] = throttle - rollCorr + pitchCorr + yawCorr; // Front LEFT  (CW)

    for (int i = 0; i < 4; i++) {
        outputs[i] = constrain(outputs[i], MOTOR_MIN, MOTOR_MAX);
    }

    // Debug: print motor outputs so tilt corrections can be verified
    // Format: M0=RR  M1=FR  M2=RL  M3=FL
    Debug::log("[Motors] RR="); Debug::log(outputs[0]);
    Debug::log(" FR=");         Debug::log(outputs[1]);
    Debug::log(" RL=");         Debug::log(outputs[2]);
    Debug::log(" FL=");         Debug::logln(outputs[3]);
}

int Mixer::getMotor(int index) const {
    if (index < 0 || index >= 4) return MOTOR_IDLE;
    return outputs[index];
}
