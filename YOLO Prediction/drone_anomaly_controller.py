#!/usr/bin/env python3
"""
=============================================================================
  AUTONOMOUS DRONE ANOMALY DETECTION & CONTROL SYSTEM
=============================================================================
  Integrates:
    - YOLOv8 helmet + vehicle violation detection
    - MQTT RC drone control (ESP32 / ESPFC)
    - Autonomous behaviors: patrol, track, hover, altitude adjust

  Drone Response to Violation:
    1. Yaw  → rotate to face the violator (horizontal centering)
    2. Pitch → move forward toward violator
    3. Altitude (throttle) adjust based on bounding box size
    4. Hover + record once centered on target

  No Violation:
    → Auto-patrol: sweep yaw left ↔ right scanning for targets

  Video Source:
    → Video file (testing). Switch cap source for live camera.

  Controls:
    E          → ARM drone
    Q          → DISARM drone
    P          → Toggle autonomous mode ON/OFF
    Ctrl+C     → Emergency stop + disarm

  Install:
    pip install paho-mqtt keyboard opencv-python cvzone ultralytics
=============================================================================
"""

import cv2
import time
import math
import threading
from datetime import datetime
import cvzone
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import keyboard

# =============================================================================
#  MQTT CONFIG
# =============================================================================
BROKER    = "192.168.43.222"   # Your PC LAN IP (Mosquitto broker)
PORT      = 1883
TOPIC     = "espfc/rc"

# =============================================================================
#  RC CHANNEL CONSTANTS
# =============================================================================
CHANNELS      = 8
RATE          = 40.0           # Hz — publish rate
CENTER        = 1500
MIN_VAL       = 1000
MAX_VAL       = 2000
THROTTLE_LOW  = 1000
THROTTLE_HOVER= 1550           # Tune this for your drone to hover
THROTTLE_MAX  = 1850

# RC step sizes
YAW_STEP      = 50             # Yaw offset from center during patrol/track
PITCH_STEP    = 30             # Forward pitch offset when approaching target
ROLL_STEP     = 30             # Not used in auto; available for manual
THROTTLE_STEP = 10             # Throttle increment per frame

# =============================================================================
#  VISION / DETECTION CONFIG
# =============================================================================
VIDEO_SOURCE   = "Media/traffic.mp4"   # Change to 0 for webcam / drone cam
FRAME_SKIP     = 3                     # Process every Nth frame
MAX_ANOMALIES  = 5                     # Max violations scored per frame

COCO_MODEL_PATH   = "yolov8n.pt"
HELMET_MODEL_PATH = "Weights/best.pt"

HELMET_CLASS_NAMES = ['With Helmet', 'Without Helmet']

COCO_VEHICLE_CLASSES = {
    0: 'Person',
    2: 'Car',
    3: 'Motorcycle',
    5: 'Bus',
    7: 'Truck'
}

SCORE_THRESHOLD = 0.02

# =============================================================================
#  DRONE CONTROLLER CONFIG
# =============================================================================
# Yaw tracking PD
YAW_Kp          = 0.06       # Proportional gain for yaw
YAW_Kd          = 0.01       # Derivative gain for yaw
YAW_DEADZONE    = 30         # px — ignore small horizontal errors
YAW_ALPHA       = 0.80       # Smoothing factor

# Altitude (throttle) tracking — based on bounding box height ratio
ALT_TARGET_RATIO   = 0.35    # Target: person occupies 35% of frame height
ALT_Kp             = 80      # Throttle adjustment per ratio error
ALT_DEADZONE       = 0.05    # Ignore small altitude errors

# Approach (pitch) — based on bounding box width ratio
APPROACH_TARGET_RATIO = 0.20 # Target: person occupies 20% of frame width
APPROACH_Kp           = 0.4  # Pitch step scale
APPROACH_DEADZONE     = 0.05

# Patrol sweep config
PATROL_SPEED          = 15   # Yaw step size during patrol
PATROL_HOLD_FRAMES    = 30   # Frames to hold at each sweep end

# Hover-lock: when yaw error < this (px), we are "locked on"
LOCK_DEADZONE         = 40

# Return to neutral after N seconds of no violation
ANOMALY_HOLD_TIME     = 5.0

# =============================================================================
#  SHARED STATE (thread-safe via lock)
# =============================================================================
class DroneState:
    def __init__(self):
        self.lock = threading.Lock()

        # RC channels
        self.arm       = 1000
        self.throttle  = THROTTLE_LOW
        self.roll      = CENTER
        self.pitch     = CENTER
        self.yaw       = CENTER

        # Mode flags
        self.autonomous = False   # Toggle with P key
        self.armed      = False

        # Tracking state
        self.target_cx    = None   # Violator center X
        self.target_ratio_h = None # Bounding box h / frame_h
        self.target_ratio_w = None # Bounding box w / frame_w
        self.last_anomaly_time = 0.0
        self.anomaly_active    = False

        # PD state
        self.prev_yaw_error   = 0.0
        self.smooth_yaw_error = 0.0

        # Patrol state
        self.patrol_direction = 1   # +1 = right, -1 = left
        self.patrol_hold      = 0

        # Status string for HUD
        self.status = "DISARMED | MANUAL"

    def update_target(self, cx, ratio_h, ratio_w):
        with self.lock:
            self.target_cx      = cx
            self.target_ratio_h = ratio_h
            self.target_ratio_w = ratio_w
            self.last_anomaly_time = time.time()
            self.anomaly_active    = True

    def clear_target(self):
        with self.lock:
            self.target_cx      = None
            self.target_ratio_h = None
            self.target_ratio_w = None

    def get_channels(self):
        with self.lock:
            vals = [CENTER] * CHANNELS
            vals[0] = int(self.roll)
            vals[1] = int(self.pitch)
            vals[2] = int(self.throttle)
            vals[3] = int(self.yaw)
            vals[4] = int(self.arm)
            return vals


state = DroneState()


# =============================================================================
#  UTILITY
# =============================================================================
def clamp(v, lo=MIN_VAL, hi=MAX_VAL):
    return max(lo, min(hi, v))


# =============================================================================
#  MQTT SETUP
# =============================================================================
def on_connect(client, userdata, flags, rc):
    print("[MQTT] Connected ✓" if rc == 0 else f"[MQTT] Failed rc={rc}")

def on_disconnect(client, userdata, rc):
    print(f"[MQTT] Disconnected rc={rc}")

mqtt_client = mqtt.Client()
mqtt_client.on_connect    = on_connect
mqtt_client.on_disconnect = on_disconnect
mqtt_client.reconnect_delay_set(min_delay=1, max_delay=5)
mqtt_client.connect(BROKER, PORT, keepalive=10)
mqtt_client.loop_start()


def publish_rc():
    vals = state.get_channels()
    mqtt_client.publish(TOPIC, ",".join(str(v) for v in vals), qos=0)


# =============================================================================
#  RC PUBLISH THREAD — runs at RATE Hz independently of vision
# =============================================================================
def rc_thread():
    interval = 1.0 / RATE
    while True:
        publish_rc()
        time.sleep(interval)

threading.Thread(target=rc_thread, daemon=True).start()


# =============================================================================
#  AUTONOMOUS CONTROL LOGIC — called every processed frame
# =============================================================================
def autonomous_control(frame_cx, frame_h, frame_w):
    """
    Updates drone RC values based on current detection state.
    Called from the vision loop after anomaly list is computed.
    """
    now = time.time()

    with state.lock:
        if not state.autonomous or not state.armed:
            return

        # ── CHECK IF ANOMALY RECENTLY SEEN ───────────────────────────
        if state.anomaly_active:
            if now - state.last_anomaly_time > ANOMALY_HOLD_TIME:
                # Lost target — return to neutral and patrol
                state.anomaly_active = False
                state.clear_target()
                state.pitch    = CENTER
                state.throttle = THROTTLE_HOVER
                state.status   = "PATROLLING"

        # ─────────────────────────────────────────────────────────────
        #  TRACKING MODE — violation detected
        # ─────────────────────────────────────────────────────────────
        if state.anomaly_active and state.target_cx is not None:
            state.status = "TRACKING VIOLATOR"

            # ── 1. YAW — rotate to face violator ──────────────────
            yaw_error = state.target_cx - frame_cx   # + = target is right

            if abs(yaw_error) < YAW_DEADZONE:
                yaw_error = 0

            # PD smoothing
            derivative = yaw_error - state.prev_yaw_error
            state.smooth_yaw_error = (YAW_ALPHA * state.smooth_yaw_error +
                                      (1 - YAW_ALPHA) * yaw_error)
            state.prev_yaw_error = state.smooth_yaw_error

            yaw_delta  = YAW_Kp * state.smooth_yaw_error + YAW_Kd * derivative
            yaw_delta  = clamp(yaw_delta, -YAW_STEP, YAW_STEP)
            state.yaw  = clamp(CENTER + yaw_delta)

            locked_on = abs(yaw_error) < LOCK_DEADZONE

            # ── 2. ALTITUDE — adjust throttle by bounding box height ──
            if state.target_ratio_h is not None:
                alt_error = ALT_TARGET_RATIO - state.target_ratio_h
                if abs(alt_error) > ALT_DEADZONE:
                    throttle_adj = ALT_Kp * alt_error
                    state.throttle = clamp(
                        state.throttle + throttle_adj,
                        THROTTLE_HOVER - 150,
                        THROTTLE_HOVER + 150
                    )

            # ── 3. PITCH — move forward if far away ───────────────
            if locked_on and state.target_ratio_w is not None:
                approach_error = APPROACH_TARGET_RATIO - state.target_ratio_w
                if approach_error > APPROACH_DEADZONE:
                    # Target is small/far — pitch forward
                    pitch_offset  = APPROACH_Kp * approach_error * PITCH_STEP
                    state.pitch   = clamp(CENTER + int(pitch_offset))
                    state.status  = "APPROACHING TARGET"
                else:
                    # Close enough — hover and record
                    state.pitch  = CENTER
                    state.status = "LOCKED & HOVERING"
            else:
                # Still rotating — don't pitch yet
                state.pitch = CENTER

        # ─────────────────────────────────────────────────────────────
        #  PATROL MODE — no violation detected
        # ─────────────────────────────────────────────────────────────
        else:
            state.pitch    = CENTER
            state.throttle = THROTTLE_HOVER
            state.status   = "PATROLLING"

            # Reset yaw PD
            state.prev_yaw_error   = 0.0
            state.smooth_yaw_error = 0.0

            # Sweep yaw left ↔ right
            if state.patrol_hold > 0:
                state.patrol_hold -= 1
                state.yaw = CENTER  # brief pause at center
            else:
                sweep_target = CENTER + state.patrol_direction * YAW_STEP * 2
                state.yaw    = clamp(sweep_target)

                # Reached sweep limit — flip direction
                if state.yaw >= CENTER + YAW_STEP * 2 - 5:
                    state.patrol_direction = -1
                    state.patrol_hold      = PATROL_HOLD_FRAMES
                elif state.yaw <= CENTER - YAW_STEP * 2 + 5:
                    state.patrol_direction = 1
                    state.patrol_hold      = PATROL_HOLD_FRAMES


# =============================================================================
#  KEYBOARD INPUT THREAD — manual arm/disarm/mode toggle
# =============================================================================
def keyboard_thread():
    print("[INPUT] Keyboard listener started.")
    while True:
        # ARM
        if keyboard.is_pressed("e"):
            with state.lock:
                if not state.armed:
                    if state.throttle > 1100:
                        print(f"  ⚠ Lower throttle first! THR={state.throttle}")
                    else:
                        state.arm   = 2000
                        state.armed = True
                        print(">>> ARMED <<<")
            time.sleep(0.3)

        # DISARM
        if keyboard.is_pressed("q"):
            with state.lock:
                if state.armed:
                    state.arm      = 1000
                    state.armed    = False
                    state.throttle = THROTTLE_LOW
                    state.pitch    = CENTER
                    state.roll     = CENTER
                    state.yaw      = CENTER
                    state.autonomous = False
                    print(">>> DISARMED <<<")
            time.sleep(0.3)

        # TOGGLE AUTONOMOUS
        if keyboard.is_pressed("p"):
            with state.lock:
                state.autonomous = not state.autonomous
                mode = "AUTONOMOUS" if state.autonomous else "MANUAL"
                if state.autonomous and state.armed:
                    state.throttle = THROTTLE_HOVER
                print(f">>> MODE: {mode} <<<")
            time.sleep(0.3)

        # MANUAL CONTROLS (only when not autonomous)
        with state.lock:
            auto = state.autonomous
            armed = state.armed

        if not auto and armed:
            with state.lock:
                if keyboard.is_pressed("up"):
                    state.throttle = clamp(state.throttle + THROTTLE_STEP,
                                           THROTTLE_LOW, THROTTLE_MAX)
                if keyboard.is_pressed("down"):
                    state.throttle = clamp(state.throttle - THROTTLE_STEP,
                                           THROTTLE_LOW, THROTTLE_MAX)
                if keyboard.is_pressed("left"):
                    state.roll = clamp(CENTER - ROLL_STEP)
                elif keyboard.is_pressed("right"):
                    state.roll = clamp(CENTER + ROLL_STEP)
                else:
                    state.roll = CENTER

                if keyboard.is_pressed("w"):
                    state.pitch = clamp(CENTER + PITCH_STEP)
                elif keyboard.is_pressed("s"):
                    state.pitch = clamp(CENTER - PITCH_STEP)
                else:
                    state.pitch = CENTER

                if keyboard.is_pressed("a"):
                    state.yaw = clamp(CENTER - YAW_STEP)
                elif keyboard.is_pressed("d"):
                    state.yaw = clamp(CENTER + YAW_STEP)
                else:
                    state.yaw = CENTER

        time.sleep(1.0 / RATE)

threading.Thread(target=keyboard_thread, daemon=True).start()


# =============================================================================
#  YOLO MODELS
# =============================================================================
print("[YOLO] Loading models...")
coco_model   = YOLO(COCO_MODEL_PATH)
helmet_model = YOLO(HELMET_MODEL_PATH)
print("[YOLO] Models loaded ✓")

# =============================================================================
#  VIDEO CAPTURE
# =============================================================================
cap = cv2.VideoCapture(VIDEO_SOURCE)
frame_w, frame_h, frame_cx = None, None, None
frame_count = 0

print("=" * 60)
print("  DRONE ANOMALY CONTROLLER")
print("  E=ARM  Q=DISARM  P=Toggle Auto  Ctrl+C=Emergency Stop")
print("  Manual (when not auto): Up/Down=Throttle  W/S=Pitch")
print("                          A/D=Yaw  Left/Right=Roll")
print("=" * 60)

# =============================================================================
#  MAIN VISION + CONTROL LOOP
# =============================================================================
try:
    while True:
        success, img = cap.read()
        if not success:
            # Loop video file
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1

        if frame_w is None:
            frame_h, frame_w = img.shape[:2]
            frame_cx = frame_w // 2
            print(f"[VIDEO] Frame size: {frame_w}x{frame_h}")

        # ── SKIP FRAMES for performance ───────────────────────────
        if frame_count % FRAME_SKIP != 0:
            # Still draw HUD on skipped frames
            pass
        else:
            anomalies = []

            # ── COCO DETECTION ────────────────────────────────────
            coco_results = coco_model(img, stream=True)

            for r in coco_results:
                for box in r.boxes:
                    cls = int(box.cls[0])

                    # ── PERSON → helmet check ──────────────────────
                    if cls == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bw, bh = x2 - x1, y2 - y1
                        if bw <= 0 or bh <= 0:
                            continue

                        person_crop    = img[y1:y2, x1:x2]
                        helmet_results = helmet_model(person_crop, stream=True)

                        helmet_status = None
                        helmet_conf   = 0

                        for h_r in helmet_results:
                            if len(h_r.boxes) > 0:
                                hb            = h_r.boxes[0]
                                helmet_status = int(hb.cls[0])
                                helmet_conf   = float(hb.conf[0])
                                break

                        if helmet_status == 1:  # WITHOUT HELMET
                            person_cx    = x1 + bw // 2
                            size_weight  = (bw * bh) / (frame_w * frame_h)
                            dist         = abs(person_cx - frame_cx)
                            pos_weight   = 1 - (dist / frame_w)

                            score = 2.0 * helmet_conf * size_weight * pos_weight

                            anomalies.append({
                                "score"   : score,
                                "cx"      : person_cx,
                                "box"     : (x1, y1, bw, bh),
                                "conf"    : helmet_conf,
                                "ratio_h" : bh / frame_h,
                                "ratio_w" : bw / frame_w,
                                "type"    : "NO HELMET"
                            })
                        else:
                            # Draw with helmet person
                            cvzone.cornerRect(img, (x1, y1, bw, bh), colorC=(0, 255, 0))
                            if helmet_status is not None:
                                label = f'{HELMET_CLASS_NAMES[helmet_status]} {helmet_conf:.2f}'
                                cvzone.putTextRect(img, label,
                                                   (x1, max(30, y1 - 10)),
                                                   colorR=(0, 255, 0))

                    # ── VEHICLES ───────────────────────────────────
                    elif cls in COCO_VEHICLE_CLASSES:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bw, bh = x2 - x1, y2 - y1
                        conf   = float(box.conf[0])
                        cvzone.cornerRect(img, (x1, y1, bw, bh), colorC=(255, 165, 0))
                        cvzone.putTextRect(
                            img,
                            f'{COCO_VEHICLE_CLASSES[cls]} {conf:.2f}',
                            (x1, max(30, y1 - 10)),
                            colorR=(255, 165, 0)
                        )

            # ── RANK & DRAW ANOMALIES ─────────────────────────────
            anomalies.sort(key=lambda x: x["score"], reverse=True)
            anomalies = anomalies[:MAX_ANOMALIES]

            for i, a in enumerate(anomalies):
                x1, y1, bw, bh = a["box"]
                color = (0, 0, 255) if i == 0 else (0, 100, 255)
                label = f"#{i+1} {a['type']} {a['score']:.2f}"
                cvzone.cornerRect(img, (x1, y1, bw, bh), colorC=color)
                cvzone.putTextRect(img, label, (x1, max(30, y1 - 10)), colorR=color)

            # ── UPDATE DRONE TARGET ───────────────────────────────
            if anomalies and anomalies[0]["score"] > SCORE_THRESHOLD:
                best = anomalies[0]
                state.update_target(
                    cx      = best["cx"],
                    ratio_h = best["ratio_h"],
                    ratio_w = best["ratio_w"]
                )
            else:
                state.clear_target()

            # ── RUN AUTONOMOUS CONTROL ────────────────────────────
            autonomous_control(frame_cx, frame_h, frame_w)

        # ── HUD OVERLAY ───────────────────────────────────────────
        with state.lock:
            s = state.status if state.autonomous else (
                "ARMED | MANUAL" if state.armed else "DISARMED | MANUAL"
            )
            thr  = state.throttle
            yw   = state.yaw
            pt   = state.pitch
            rl   = state.roll
            auto = state.autonomous
            armed = state.armed

        hud_color = (0, 255, 0) if armed else (0, 0, 200)
        auto_str  = "AUTO" if auto else "MANUAL"

        cv2.rectangle(img, (0, 0), (420, 130), (0, 0, 0), -1)
        cv2.putText(img, f"STATUS : {s}",         (10,  22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)
        cv2.putText(img, f"MODE   : {auto_str}",  (10,  46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        cv2.putText(img, f"THR={thr}  YAW={yw}",  (10,  70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, f"PITCH={pt}  ROLL={rl}", (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "P=Auto  E=Arm  Q=Disarm  Q=Quit", (10, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        cv2.imshow("Drone Anomaly Controller", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[!] Emergency stop triggered.")

finally:
    print("Sending disarm burst...")
    with state.lock:
        state.arm      = 1000
        state.throttle = THROTTLE_LOW
        state.pitch    = CENTER
        state.roll     = CENTER
        state.yaw      = CENTER

    for _ in range(15):
        publish_rc()
        time.sleep(0.05)

    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    cap.release()
    cv2.destroyAllWindows()
    print("Shutdown complete.")
