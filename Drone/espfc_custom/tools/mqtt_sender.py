#!/usr/bin/env python3
"""
mqtt_send.py

MQTT RC sender aligned with ESP-FC firmware:
- Channels: [ROLL, PITCH, THROTTLE, YAW, ARM, AUX2, AUX3, AUX4]
- Values: 1000–2000 (center = 1500)
- ARM toggle: 'E' to ARM (2000), 'Q' to DISARM (1000)
- Arrow keys + WASD control throttle/roll/pitch/yaw
"""

import time
import paho.mqtt.client as mqtt
import keyboard  # pip install keyboard

BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "espfc/rc"
CHANNELS = 8
RATE = 100.0  # Hz 

def publish(client, values):
    payload = ','.join(str(int(v)) for v in values)
    client.publish(TOPIC, payload)

    # Print with labels for clarity
    print(
        f"ROLL={values[0]} | "
        f"PITCH={values[1]} | "
        f"THROTTLE={values[2]} | "
        f"YAW={values[3]} | "
        f"ARM={values[4]} | "
        f"AUX2={values[5]} | AUX3={values[6]} | AUX4={values[7]}"
    )

def main():
    client = mqtt.Client()
    client.connect(BROKER, PORT, 60)
    client.loop_start()
    print(f"Connected to {BROKER}:{PORT}, publishing to {TOPIC}")

    aux1_state = 1000  # ARM channel starts DISARMED

    try:
        while True:
            # Default neutral values
            vals = [1500] * CHANNELS
            vals[2] = 1100   # throttle low by default
            vals[4] = aux1_state  # ARM channel

            # Toggle ARM
            if keyboard.is_pressed("e"):
                aux1_state = 2000
                print("ARM sent")
            if keyboard.is_pressed("q"):
                aux1_state = 1000
                print("DISARM sent")

            # Arrow keys for throttle/roll
            if keyboard.is_pressed("up"):
                vals[2] = 1990  # throttle up
            if keyboard.is_pressed("down"):
                vals[2] = 1200  # throttle down
            if keyboard.is_pressed("left"):
                vals[0] = 1200  # roll left
            if keyboard.is_pressed("right"):
                vals[0] = 1800  # roll right

            # WASD for pitch/yaw
            if keyboard.is_pressed("w"):
                vals[1] = 1800  # pitch forward
            if keyboard.is_pressed("s"):
                vals[1] = 1200  # pitch backward
            if keyboard.is_pressed("a"):
                vals[3] = 1200  # yaw left
            if keyboard.is_pressed("d"):
                vals[3] = 1800  # yaw right

            publish(client, vals)
            time.sleep(1.0 / RATE)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
