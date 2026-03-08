#include "Mixer.h"
#include "Config.h"
#include "Debug.h"

// include PID corrections
void Mixer::update(RcInput& rc, float rollCorr, float pitchCorr, float yawCorr)
{
    int throttle = rc.get(THROTTLE_CH);
    bool armed   = rc.get(ARM_CH) > 1500;

    if (!armed)
    {
        for(int i=0;i<4;i++)
            outputs[i] = MOTOR_MIN;
        return;
    }

    // ensure idle spin
    throttle = max(throttle, MOTOR_IDLE);

    // QUADX mix
    outputs[0] = throttle + rollCorr + pitchCorr - yawCorr; // M1 Rear Left
    outputs[1] = throttle + rollCorr - pitchCorr + yawCorr; // M2 Front Left
    outputs[2] = throttle - rollCorr + pitchCorr + yawCorr; // M3 Rear Right
    outputs[3] = throttle - rollCorr - pitchCorr - yawCorr; // M4 Front Right

    for(int i=0;i<4;i++)
    {
        outputs[i] = constrain(outputs[i], MOTOR_MIN, MOTOR_MAX);
    }
}

int Mixer::getMotor(int index) const {
    if (index < 0 || index >= 4) return MOTOR_IDLE;
    return outputs[index];
}
