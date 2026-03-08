#!/usr/bin/env python3

import time
import paho.mqtt.client as mqtt
import keyboard

BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "espfc/rc"

CHANNELS = 8
RATE = 100.0

CENTER = 1510        # slight trim for stability
MIN_VAL = 1000
MAX_VAL = 2000

ROLL_STEP = 300
PITCH_STEP = 300
YAW_STEP = 300

THROTTLE_LOW = 1100
THROTTLE_HIGH = 1800


def publish(client, values):
    payload = ",".join(str(int(v)) for v in values)
    client.publish(TOPIC, payload)

    print(
        f"R={values[0]} "
        f"P={values[1]} "
        f"T={values[2]} "
        f"Y={values[3]} "
        f"ARM={values[4]}"
    )


def clamp(v):
    return max(MIN_VAL, min(MAX_VAL, v))


def main():

    client = mqtt.Client()
    client.connect(BROKER, PORT, 60)
    client.loop_start()

    print("MQTT RC sender started")

    arm_state = 1000

    try:

        while True:

            vals = [CENTER] * CHANNELS

            # throttle default
            vals[2] = THROTTLE_LOW

            # arm channel
            vals[4] = arm_state

            # ARM / DISARM
            if keyboard.is_pressed("e"):
                arm_state = 2000
                print("ARM")

            if keyboard.is_pressed("q"):
                arm_state = 1000
                print("DISARM")

            # THROTTLE
            if keyboard.is_pressed("up"):
                vals[2] = THROTTLE_HIGH

            if keyboard.is_pressed("down"):
                vals[2] = THROTTLE_LOW

            # ROLL
            if keyboard.is_pressed("left"):
                vals[0] = clamp(CENTER - ROLL_STEP)

            if keyboard.is_pressed("right"):
                vals[0] = clamp(CENTER + ROLL_STEP)

            # PITCH
            if keyboard.is_pressed("w"):
                vals[1] = clamp(CENTER + PITCH_STEP)

            if keyboard.is_pressed("s"):
                vals[1] = clamp(CENTER - PITCH_STEP)

            # YAW
            if keyboard.is_pressed("a"):
                vals[3] = clamp(CENTER - YAW_STEP)

            if keyboard.is_pressed("d"):
                vals[3] = clamp(CENTER + YAW_STEP)

            publish(client, vals)

            time.sleep(1.0 / RATE)

    except KeyboardInterrupt:
        print("Stopped")

    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()