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

    // QUADX mix
    outputs[0] = throttle + rollCorr + pitchCorr - yawCorr; // M1 Rear Left
    outputs[1] = throttle + rollCorr - pitchCorr + yawCorr; // M2 Front Left
    outputs[2] = throttle - rollCorr + pitchCorr + yawCorr; // M3 Rear Right
    outputs[3] = throttle - rollCorr - pitchCorr - yawCorr; // M4 Front Right

    for (int i = 0; i < 4; i++) {
        outputs[i] = constrain(outputs[i], MOTOR_MIN, MOTOR_MAX);
    }
}

int Mixer::getMotor(int index) const {
    if (index < 0 || index >= 4) return MOTOR_IDLE;
    return outputs[index];
}
