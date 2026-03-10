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

Serial logging:
  When DEBUG_SERIAL=1 is set in firmware Config.h, this script
  automatically reads the ESP32 serial port and writes every line
  to logs/drone_YYYYMMDD_HHMMSS.log in the tools/ folder.
  Set SERIAL_PORT below to your ESP32 COM port (e.g. "COM3").
  Set SERIAL_PORT = None to auto-detect.
  Set SERIAL_LOGGING = False to disable entirely.
"""

import time
import os
import threading
from datetime import datetime
import paho.mqtt.client as mqtt
import keyboard

# ── MQTT ──────────────────────────────────────────────────────
BROKER        = "192.168.43.222"   # Your PC's LAN IP (local Mosquitto broker)
PORT          = 1883
TOPIC         = "espfc/rc"

# ── RC ────────────────────────────────────────────────────────
CHANNELS      = 8
RATE          = 40.0
CENTER        = 1500
MIN_VAL       = 1000
MAX_VAL       = 2000
ROLL_STEP     = 100
PITCH_STEP    = 100
YAW_STEP      = 100
THROTTLE_STEP = 10
THROTTLE_LOW  = 1000
THROTTLE_MAX  = 1800

# ── SERIAL LOGGING ────────────────────────────────────────────
SERIAL_LOGGING = False       # Set False to disable
SERIAL_PORT    = "COM4"       # None = auto-detect, or set e.g. "COM3"
SERIAL_BAUD    = 115200


# ── Serial logger ─────────────────────────────────────────────

def find_esp32_port():
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            desc = (p.description or "").lower()
            if any(k in desc for k in ["cp210", "ch340", "ch341", "uart", "esp"]):
                return p.device
        return ports[0].device if ports else None
    except Exception:
        return None


def serial_logger_thread(port, baud, log_path, stop_event):
    try:
        import serial
    except ImportError:
        print("  [LOG] pyserial not installed — run: pip install pyserial")
        return
    try:
        ser = serial.Serial(port, baud, timeout=1)
        print(f"  [LOG] Serial opened: {port}")
    except Exception as e:
        print(f"  [LOG] Cannot open {port}: {e}")
        return

    with open(log_path, "a", encoding="utf-8") as f:
        while not stop_event.is_set():
            try:
                raw = ser.readline()
                if not raw:
                    continue
                line  = raw.decode("utf-8", errors="replace").rstrip()
                ts    = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                f.write(f"[{ts}] {line}\n")
                f.flush()
            except Exception:
                break
    ser.close()


def start_serial_logging(port):
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"drone_{ts}.log")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# ESP32 Drone Serial Log\n")
        f.write(f"# Session : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Port    : {port} @ {SERIAL_BAUD}\n")
        f.write(f"# {'=' * 48}\n\n")

    stop_event = threading.Event()
    threading.Thread(
        target=serial_logger_thread,
        args=(port, SERIAL_BAUD, log_path, stop_event),
        daemon=True
    ).start()
    print(f"  [LOG] → logs/{os.path.basename(log_path)}")
    return stop_event, log_path


# ── MQTT callbacks ────────────────────────────────────────────

def publish(client, values):
    client.publish(TOPIC, ",".join(str(int(v)) for v in values), qos=0)

def on_connect(client, userdata, flags, rc):
    print("[MQTT] Connected ✓" if rc == 0 else f"[MQTT] Failed rc={rc}")

def on_disconnect(client, userdata, rc):
    print(f"[MQTT] Disconnected rc={rc} — reconnecting...")

def clamp(v, lo=MIN_VAL, hi=MAX_VAL):
    return max(lo, min(hi, v))


# ── Main ──────────────────────────────────────────────────────

def main():
    # Serial logging
    stop_event = None
    log_path   = None

    if SERIAL_LOGGING:
        port = SERIAL_PORT or find_esp32_port()
        if port:
            stop_event, log_path = start_serial_logging(port)
        else:
            print("  [LOG] No COM port found. Set SERIAL_PORT='COMx' in script.")

    # MQTT
    client = mqtt.Client()
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.reconnect_delay_set(min_delay=1, max_delay=5)
    client.connect(BROKER, PORT, keepalive=10)
    client.loop_start()

    print("=" * 50)
    print("  MQTT RC Sender  (Windows)")
    print("  E=ARM    Q=DISARM    Ctrl+C=Quit")
    print("  Up/Down=Throttle    Left/Right=Roll")
    print("  W/S=Pitch           A/D=Yaw")
    if log_path:
        print(f"  Log → logs/{os.path.basename(log_path)}")
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
            if keyboard.is_pressed("e"):
                if arm_state != 2000:
                    if throttle_state > 1100:
                        print(f"  ⚠ Lower throttle first! THR={throttle_state} (needs ≤ 1100)")
                    else:
                        arm_state = 2000
                        print(">>> ARMED <<<")
            if keyboard.is_pressed("q"):
                if arm_state != 1000:
                    arm_state      = 1000
                    throttle_state = THROTTLE_LOW
                    print(">>> DISARMED <<<")

            if keyboard.is_pressed("up"):
                throttle_state = clamp(throttle_state + THROTTLE_STEP, THROTTLE_LOW, THROTTLE_MAX)
            if keyboard.is_pressed("down"):
                throttle_state = clamp(throttle_state - THROTTLE_STEP, THROTTLE_LOW, THROTTLE_MAX)

            if keyboard.is_pressed("left"):
                roll_state = clamp(CENTER - ROLL_STEP)
            elif keyboard.is_pressed("right"):
                roll_state = clamp(CENTER + ROLL_STEP)
            else:
                roll_state = CENTER

            if keyboard.is_pressed("w"):
                pitch_state = clamp(CENTER + PITCH_STEP)
            elif keyboard.is_pressed("s"):
                pitch_state = clamp(CENTER - PITCH_STEP)
            else:
                pitch_state = CENTER

            if keyboard.is_pressed("a"):
                yaw_state = clamp(CENTER - YAW_STEP)
            elif keyboard.is_pressed("d"):
                yaw_state = clamp(CENTER + YAW_STEP)
            else:
                yaw_state = CENTER

            vals = [CENTER] * CHANNELS
            vals[0] = roll_state
            vals[1] = pitch_state
            vals[2] = throttle_state
            vals[3] = yaw_state
            vals[4] = arm_state
            publish(client, vals)

            now    = time.time()
            gap_ms = (now - last_sent) * 1000
            if gap_ms > 500:
                print(f"  ⚠ Send gap {gap_ms:.0f}ms")
            last_sent = now

            if frame % int(RATE) == 0:
                armed_str = "ARMED  " if arm_state > 1500 else "DISARMED"
                print(f"[{armed_str}]  THR={throttle_state}  "
                      f"ROLL={roll_state}  PITCH={pitch_state}  YAW={yaw_state}")
            frame += 1
            time.sleep(1.0 / RATE)

    except KeyboardInterrupt:
        print("\nStopping — sending disarm burst...")

    finally:
        vals = [CENTER] * CHANNELS
        vals[2] = THROTTLE_LOW
        vals[4] = 1000
        for _ in range(10):
            publish(client, vals)
            time.sleep(0.05)
        client.loop_stop()
        client.disconnect()
        if stop_event:
            stop_event.set()
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n# Session end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"Log saved: {log_path}")
        print("Disconnected.")


if __name__ == "__main__":
    main()
