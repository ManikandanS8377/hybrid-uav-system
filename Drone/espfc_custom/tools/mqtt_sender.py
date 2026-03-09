#!/usr/bin/env python3
"""
MQTT RC Sender for ESP32 Drone (Windows)
Uses the 'keyboard' library — works on Windows without admin rights.

Controls:
  E           → ARM   (hold or tap)
  Q           → DISARM
  Up / Down   → Throttle increase / decrease
  Left / Right→ Roll  (self-centers on release)
  W / S       → Pitch (self-centers on release)
  A / D       → Yaw   (self-centers on release)
  Ctrl+C      → Quit  (auto-disarm burst sent)
"""

import time
import paho.mqtt.client as mqtt
import keyboard

BROKER        = "192.168.43.222"   # ← CHANGE to your PC's LAN IP, e.g. "192.168.1.100"
                                    #   Run tools/setup_local_broker.bat first
PORT          = 1883
TOPIC         = "espfc/rc"

CHANNELS      = 8
RATE          = 40.0    # 40Hz — slightly slower gives broker more headroom, reduces drops

CENTER        = 1500    # aligned with firmware deadband center
MIN_VAL       = 1000
MAX_VAL       = 2000

ROLL_STEP     = 100     # (center±100-1500)*0.2 = ±20 deg/s — safe for brushed
PITCH_STEP    = 100
YAW_STEP      = 100
THROTTLE_STEP = 10
THROTTLE_LOW  = 1000    # canArm() requires throttle <= 1100, start at 1000
THROTTLE_MAX  = 1800    # soft ceiling for brushed mini drone


def publish(client, values):
    payload = ",".join(str(int(v)) for v in values)
    client.publish(TOPIC, payload, qos=0)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[MQTT] Connected ✓")
    else:
        print(f"[MQTT] Connect failed rc={rc}")

def on_disconnect(client, userdata, rc):
    print(f"[MQTT] Disconnected rc={rc} — auto-reconnecting...")


def clamp(v, lo=MIN_VAL, hi=MAX_VAL):
    return max(lo, min(hi, v))


def main():
    client = mqtt.Client()
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.reconnect_delay_set(min_delay=1, max_delay=5)  # auto-reconnect on drop
    client.connect(BROKER, PORT, keepalive=10)  # 10s keepalive — detect drops faster
    client.loop_start()

    print("=" * 50)
    print("MQTT RC Sender started (Windows)")
    print("E=ARM  Q=DISARM")
    print("Up/Down=Throttle  Left/Right=Roll")
    print("W/S=Pitch  A/D=Yaw  Ctrl+C=Quit")
    print("=" * 50)

    arm_state      = 1000
    throttle_state = THROTTLE_LOW
    roll_state     = CENTER
    pitch_state    = CENTER
    yaw_state      = CENTER
    frame          = 0
    last_sent      = time.time()

    try:
        while True:
            # ── ARM / DISARM ──────────────────────────────
            if keyboard.is_pressed("e"):
                if arm_state != 2000:
                    if throttle_state > 1100:
                        print(f"  ⚠ Lower throttle first! THR={throttle_state} (needs ≤ 1100). Press Down arrow.")
                    else:
                        arm_state = 2000
                        print(">>> ARMED <<<")
            if keyboard.is_pressed("q"):
                if arm_state != 1000:
                    arm_state      = 1000
                    throttle_state = THROTTLE_LOW
                    print(">>> DISARMED <<<")

            # ── THROTTLE (incremental, capped at THROTTLE_MAX) ───
            if keyboard.is_pressed("up"):
                throttle_state = clamp(throttle_state + THROTTLE_STEP, THROTTLE_LOW, THROTTLE_MAX)
            if keyboard.is_pressed("down"):
                throttle_state = clamp(throttle_state - THROTTLE_STEP, THROTTLE_LOW, THROTTLE_MAX)

            # ── ROLL (self-centering) ─────────────────────
            if keyboard.is_pressed("left"):
                roll_state = clamp(CENTER - ROLL_STEP)
            elif keyboard.is_pressed("right"):
                roll_state = clamp(CENTER + ROLL_STEP)
            else:
                roll_state = CENTER

            # ── PITCH (self-centering) ────────────────────
            if keyboard.is_pressed("w"):
                pitch_state = clamp(CENTER + PITCH_STEP)
            elif keyboard.is_pressed("s"):
                pitch_state = clamp(CENTER - PITCH_STEP)
            else:
                pitch_state = CENTER

            # ── YAW (self-centering) ──────────────────────
            if keyboard.is_pressed("a"):
                yaw_state = clamp(CENTER - YAW_STEP)
            elif keyboard.is_pressed("d"):
                yaw_state = clamp(CENTER + YAW_STEP)
            else:
                yaw_state = CENTER

            # ── BUILD + SEND packet ───────────────────────
            # Channel order must match Config.h:
            # CH0=ROLL  CH1=PITCH  CH2=THROTTLE  CH3=YAW  CH4=ARM
            vals = [CENTER] * CHANNELS
            vals[0] = roll_state
            vals[1] = pitch_state
            vals[2] = throttle_state
            vals[3] = yaw_state
            vals[4] = arm_state

            publish(client, vals)

            # Warn if actual send gap exceeded 500ms (broker delay or CPU stall)
            now = time.time()
            gap_ms = (now - last_sent) * 1000
            if gap_ms > 500:
                print(f"  ⚠ Send gap {gap_ms:.0f}ms — broker may be slow!")
            last_sent = now

            # Print status once per second
            if frame % int(RATE) == 0:
                armed_str = "ARMED  " if arm_state > 1500 else "DISARMED"
                print(f"[{armed_str}]  THR={throttle_state}  "
                      f"ROLL={roll_state}  PITCH={pitch_state}  YAW={yaw_state}")
            frame += 1

            time.sleep(1.0 / RATE)

    except KeyboardInterrupt:
        print("\nStopping — sending disarm burst...")

    finally:
        # Safe exit: 10 disarm frames before disconnect
        vals = [CENTER] * CHANNELS
        vals[2] = THROTTLE_LOW
        vals[4] = 1000
        for _ in range(10):
            publish(client, vals)
            time.sleep(0.05)
        client.loop_stop()
        client.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
