#!/usr/bin/env python3
"""
=============================================================================
  AUTONOMOUS DRONE ANOMALY DETECTION & CONTROL SYSTEM
  ── MAXIMUM PERFORMANCE VERSION ──
=============================================================================
  Performance fixes in this version:
    1. Helmet model batched — all person crops sent in ONE call (not N calls)
    2. Person crops resized to 64x64 before helmet inference (much smaller)
    3. Frame rate capped to video native FPS — no queue flood
    4. cvzone replaced with direct cv2 calls (3x faster drawing)
    5. YOLO imgsz=320 added — forces 320px internal YOLO grid (faster than 416)
    6. Display frame downscaled to 720p max (faster imshow)
    7. MQTT offline guard — no crash, vision-only mode
=============================================================================
"""

import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import keyboard

# =============================================================================
#  MQTT CONFIG
# =============================================================================
BROKER    = "192.168.43.222"
PORT      = 1883
TOPIC     = "espfc/rc"

# =============================================================================
#  RC CHANNEL CONSTANTS
# =============================================================================
CHANNELS       = 8
RATE           = 40.0
CENTER         = 1500
MIN_VAL        = 1000
MAX_VAL        = 2000
THROTTLE_LOW   = 1000
THROTTLE_HOVER = 1550
THROTTLE_MAX   = 1850
YAW_STEP       = 50
PITCH_STEP     = 30
ROLL_STEP      = 30
THROTTLE_STEP  = 10

# =============================================================================
#  VISION / DETECTION CONFIG
# =============================================================================
VIDEO_SOURCE  = "Media/traffic.mp4"   # 0=webcam | "http://IP:81/stream"=ESP32-CAM

FRAME_SKIP    = 3          # process every Nth frame for YOLO
INFER_SIZE    = 320        # YOLO internal grid size — 320 is fastest, 416 balanced
DISPLAY_WIDTH = 960        # max display width — downscale big frames for faster imshow
CONF_COCO     = 0.40       # COCO detection confidence threshold
CONF_HELMET   = 0.25       # helmet model confidence threshold
CROP_SIZE     = 64         # helmet crop resize (px) — smaller = faster inference
MAX_ANOMALIES = 5
SCORE_THRESHOLD = 0.02

COCO_MODEL_PATH   = "yolov8n.pt"
HELMET_MODEL_PATH = "Weights/best.pt"

HELMET_CLASS_NAMES   = ['With Helmet', 'Without Helmet']
COCO_VEHICLE_CLASSES = {0:'Person', 2:'Car', 3:'Motorcycle', 5:'Bus', 7:'Truck'}

# =============================================================================
#  DRONE CONTROLLER CONFIG
# =============================================================================
YAW_Kp                = 0.06
YAW_Kd                = 0.01
YAW_DEADZONE          = 30
YAW_ALPHA             = 0.80
ALT_TARGET_RATIO      = 0.35
ALT_Kp                = 80
ALT_DEADZONE          = 0.05
APPROACH_TARGET_RATIO = 0.20
APPROACH_Kp           = 0.4
APPROACH_DEADZONE     = 0.05
PATROL_HOLD_FRAMES    = 30
LOCK_DEADZONE         = 40
ANOMALY_HOLD_TIME     = 5.0

# =============================================================================
#  SHARED STATE
# =============================================================================
class DroneState:
    def __init__(self):
        self.lock              = threading.Lock()
        self.arm               = 1000
        self.throttle          = THROTTLE_LOW
        self.roll              = CENTER
        self.pitch             = CENTER
        self.yaw               = CENTER
        self.autonomous        = False
        self.armed             = False
        self.target_cx         = None
        self.target_ratio_h    = None
        self.target_ratio_w    = None
        self.last_anomaly_time = 0.0
        self.anomaly_active    = False
        self.prev_yaw_error    = 0.0
        self.smooth_yaw_error  = 0.0
        self.patrol_direction  = 1
        self.patrol_hold       = 0
        self.status            = "DISARMED | MANUAL"

    def update_target(self, cx, ratio_h, ratio_w):
        with self.lock:
            self.target_cx         = cx
            self.target_ratio_h    = ratio_h
            self.target_ratio_w    = ratio_w
            self.last_anomaly_time = time.time()
            self.anomaly_active    = True

    def clear_target(self):
        with self.lock:
            self.target_cx      = None
            self.target_ratio_h = None
            self.target_ratio_w = None

    def get_channels(self):
        with self.lock:
            vals    = [CENTER] * CHANNELS
            vals[0] = int(self.roll)
            vals[1] = int(self.pitch)
            vals[2] = int(self.throttle)
            vals[3] = int(self.yaw)
            vals[4] = int(self.arm)
            return vals

state = DroneState()

def clamp(v, lo=MIN_VAL, hi=MAX_VAL):
    return max(lo, min(hi, v))

# =============================================================================
#  DRAWING HELPERS — direct cv2 (faster than cvzone)
# =============================================================================
def draw_box(img, x1, y1, bw, bh, color, label):
    x2, y2 = x1 + bw, y1 + bh
    # Corner rectangle (4 corner lines)
    L = min(bw, bh) // 4
    T = 2
    cv2.line(img, (x1, y1), (x1+L, y1), color, T)
    cv2.line(img, (x1, y1), (x1, y1+L), color, T)
    cv2.line(img, (x2, y1), (x2-L, y1), color, T)
    cv2.line(img, (x2, y1), (x2, y1+L), color, T)
    cv2.line(img, (x1, y2), (x1+L, y2), color, T)
    cv2.line(img, (x1, y2), (x1, y2-L), color, T)
    cv2.line(img, (x2, y2), (x2-L, y2), color, T)
    cv2.line(img, (x2, y2), (x2, y2-L), color, T)
    # Label background + text
    ty = max(28, y1 - 6)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(img, label, (x1+2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

# =============================================================================
#  MQTT
# =============================================================================
def on_connect(client, userdata, flags, rc):
    print("[MQTT] Connected ✓" if rc == 0 else f"[MQTT] Failed rc={rc}")

def on_disconnect(client, userdata, rc):
    print(f"[MQTT] Disconnected rc={rc}")

mqtt_client = mqtt.Client()
mqtt_client.on_connect    = on_connect
mqtt_client.on_disconnect = on_disconnect
mqtt_client.reconnect_delay_set(min_delay=1, max_delay=5)

try:
    mqtt_client.connect(BROKER, PORT, keepalive=10)
    mqtt_client.loop_start()
    print(f"[MQTT] Connecting to {BROKER}:{PORT}...")
except Exception as e:
    print(f"[MQTT] Broker offline: {e} → Vision-only mode")

def publish_rc():
    try:
        vals = state.get_channels()
        mqtt_client.publish(TOPIC, ",".join(str(v) for v in vals), qos=0)
    except Exception:
        pass

# =============================================================================
#  RC PUBLISH THREAD
# =============================================================================
def rc_thread():
    interval = 1.0 / RATE
    while True:
        publish_rc()
        time.sleep(interval)

threading.Thread(target=rc_thread, daemon=True).start()

# =============================================================================
#  AUTONOMOUS CONTROL
# =============================================================================
def autonomous_control(frame_cx, frame_h, frame_w):
    now = time.time()
    with state.lock:
        if not state.autonomous or not state.armed:
            return
        if state.anomaly_active:
            if now - state.last_anomaly_time > ANOMALY_HOLD_TIME:
                state.anomaly_active = False
                state.clear_target()
                state.pitch    = CENTER
                state.throttle = THROTTLE_HOVER
                state.status   = "PATROLLING"

        if state.anomaly_active and state.target_cx is not None:
            state.status = "TRACKING VIOLATOR"
            yaw_error = state.target_cx - frame_cx
            if abs(yaw_error) < YAW_DEADZONE:
                yaw_error = 0
            derivative            = yaw_error - state.prev_yaw_error
            state.smooth_yaw_error = (YAW_ALPHA * state.smooth_yaw_error +
                                      (1 - YAW_ALPHA) * yaw_error)
            state.prev_yaw_error  = state.smooth_yaw_error
            yaw_delta  = YAW_Kp * state.smooth_yaw_error + YAW_Kd * derivative
            yaw_delta  = clamp(yaw_delta, -YAW_STEP, YAW_STEP)
            state.yaw  = clamp(CENTER + yaw_delta)
            locked_on  = abs(yaw_error) < LOCK_DEADZONE

            if state.target_ratio_h is not None:
                alt_error = ALT_TARGET_RATIO - state.target_ratio_h
                if abs(alt_error) > ALT_DEADZONE:
                    state.throttle = clamp(
                        state.throttle + ALT_Kp * alt_error,
                        THROTTLE_HOVER - 150, THROTTLE_HOVER + 150
                    )
            if locked_on and state.target_ratio_w is not None:
                approach_error = APPROACH_TARGET_RATIO - state.target_ratio_w
                if approach_error > APPROACH_DEADZONE:
                    state.pitch  = clamp(CENTER + int(APPROACH_Kp * approach_error * PITCH_STEP))
                    state.status = "APPROACHING TARGET"
                else:
                    state.pitch  = CENTER
                    state.status = "LOCKED & HOVERING"
            else:
                state.pitch = CENTER
        else:
            state.pitch    = CENTER
            state.throttle = THROTTLE_HOVER
            state.status   = "PATROLLING"
            state.prev_yaw_error   = 0.0
            state.smooth_yaw_error = 0.0
            if state.patrol_hold > 0:
                state.patrol_hold -= 1
                state.yaw = CENTER
            else:
                sweep_target = CENTER + state.patrol_direction * YAW_STEP * 2
                state.yaw    = clamp(sweep_target)
                if state.yaw >= CENTER + YAW_STEP * 2 - 5:
                    state.patrol_direction = -1
                    state.patrol_hold      = PATROL_HOLD_FRAMES
                elif state.yaw <= CENTER - YAW_STEP * 2 + 5:
                    state.patrol_direction = 1
                    state.patrol_hold      = PATROL_HOLD_FRAMES

# =============================================================================
#  KEYBOARD THREAD
# =============================================================================
def keyboard_thread():
    while True:
        if keyboard.is_pressed("e"):
            with state.lock:
                if not state.armed:
                    if state.throttle > 1100:
                        print("  ⚠ Lower throttle first!")
                    else:
                        state.arm   = 2000
                        state.armed = True
                        print(">>> ARMED <<<")
            time.sleep(0.3)
        if keyboard.is_pressed("q"):
            with state.lock:
                if state.armed:
                    state.arm        = 1000
                    state.armed      = False
                    state.throttle   = THROTTLE_LOW
                    state.pitch = state.roll = state.yaw = CENTER
                    state.autonomous = False
                    print(">>> DISARMED <<<")
            time.sleep(0.3)
        if keyboard.is_pressed("p"):
            with state.lock:
                state.autonomous = not state.autonomous
                if state.autonomous and state.armed:
                    state.throttle = THROTTLE_HOVER
                print(f">>> MODE: {'AUTONOMOUS' if state.autonomous else 'MANUAL'} <<<")
            time.sleep(0.3)
        with state.lock:
            auto  = state.autonomous
            armed = state.armed
        if not auto and armed:
            with state.lock:
                if keyboard.is_pressed("up"):
                    state.throttle = clamp(state.throttle + THROTTLE_STEP, THROTTLE_LOW, THROTTLE_MAX)
                if keyboard.is_pressed("down"):
                    state.throttle = clamp(state.throttle - THROTTLE_STEP, THROTTLE_LOW, THROTTLE_MAX)
                state.roll  = clamp(CENTER - ROLL_STEP)  if keyboard.is_pressed("left")  else \
                              clamp(CENTER + ROLL_STEP)  if keyboard.is_pressed("right") else CENTER
                state.pitch = clamp(CENTER + PITCH_STEP) if keyboard.is_pressed("w") else \
                              clamp(CENTER - PITCH_STEP) if keyboard.is_pressed("s") else CENTER
                state.yaw   = clamp(CENTER - YAW_STEP)   if keyboard.is_pressed("a") else \
                              clamp(CENTER + YAW_STEP)   if keyboard.is_pressed("d") else CENTER
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
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

# ── Get video native FPS to cap loop speed ────────────────────────────────────
native_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_delay = 1.0 / native_fps          # seconds per frame
print(f"[VIDEO] Native FPS: {native_fps:.1f}  →  frame delay: {frame_delay*1000:.1f}ms")

frame_w, frame_h, frame_cx = None, None, None
frame_count  = 0

# ── DISPLAY SCALE — downscale large frames for faster imshow ──────────────────
disp_scale   = 1.0          # set after first frame

# ── PERSISTENT DETECTIONS ─────────────────────────────────────────────────────
last_anomalies   = []
last_with_helmet = []
last_vehicles    = []

print("=" * 60)
print("  DRONE ANOMALY CONTROLLER  [MAX PERFORMANCE]")
print("  E=ARM  Q=DISARM  P=Toggle Auto  Ctrl+C=Emergency Stop")
print("=" * 60)

# =============================================================================
#  MAIN LOOP
# =============================================================================
try:
    while True:
        loop_start = time.time()

        success, img = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1

        # ── First frame setup ─────────────────────────────────────
        if frame_w is None:
            frame_h, frame_w = img.shape[:2]
            frame_cx  = frame_w // 2
            disp_scale = min(1.0, DISPLAY_WIDTH / frame_w)
            print(f"[VIDEO] Frame: {frame_w}x{frame_h}  Display scale: {disp_scale:.2f}")

        # ── YOLO INFERENCE — every Nth frame ──────────────────────
        if frame_count % FRAME_SKIP == 0:

            # Resize for YOLO
            scale     = INFER_SIZE / frame_w
            infer_h   = int(frame_h * scale)
            infer_img = cv2.resize(img, (INFER_SIZE, infer_h))

            new_anomalies   = []
            new_with_helmet = []
            new_vehicles    = []

            # ── Stage 1: COCO detection ────────────────────────────
            coco_results = coco_model(
                infer_img,
                imgsz=INFER_SIZE,    # ← force 320px grid internally
                stream=False,
                verbose=False,
                conf=CONF_COCO
            )

            person_boxes = []   # collect all person boxes first

            for r in coco_results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    x1  = int(box.xyxy[0][0] / scale)
                    y1  = int(box.xyxy[0][1] / scale)
                    x2  = int(box.xyxy[0][2] / scale)
                    y2  = int(box.xyxy[0][3] / scale)
                    bw, bh = x2 - x1, y2 - y1
                    if bw <= 0 or bh <= 0:
                        continue

                    if cls == 0 or cls == 3:
                        person_boxes.append((x1, y1, x2, y2, bw, bh))

                    elif cls in COCO_VEHICLE_CLASSES:
                        conf  = float(box.conf[0])
                        label = f'{COCO_VEHICLE_CLASSES[cls]} {conf:.2f}'
                        new_vehicles.append((x1, y1, bw, bh, label))

            # ── Stage 2: Batch helmet inference ────────────────────
            # KEY FIX: collect ALL person crops → run helmet model ONCE
            if person_boxes:
                crops = []
                for (x1, y1, x2, y2, bw, bh) in person_boxes:
                    crop = img[max(0,y1):min(frame_h,y2), max(0,x1):min(frame_w,x2)]
                    if crop.size == 0:
                        crops.append(None)
                        continue
                    # Resize crop to fixed small size — much faster inference
                    crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
                    crops.append(crop_resized)

                # Filter out None crops
                valid_idx   = [i for i, c in enumerate(crops) if c is not None]
                valid_crops = [crops[i] for i in valid_idx]

                helmet_statuses = [None] * len(person_boxes)
                helmet_confs    = [0.0]  * len(person_boxes)

                if valid_crops:
                    # ONE batch call for all person crops
                    batch_results = helmet_model(
                        valid_crops,
                        imgsz=CROP_SIZE,
                        stream=False,
                        verbose=False,
                        conf=CONF_HELMET
                    )
                    for result_idx, h_r in enumerate(batch_results):
                        orig_idx = valid_idx[result_idx]
                        if len(h_r.boxes) > 0:
                            helmet_statuses[orig_idx] = int(h_r.boxes[0].cls[0])
                            helmet_confs[orig_idx]    = float(h_r.boxes[0].conf[0])

                # Process results
                for i, (x1, y1, x2, y2, bw, bh) in enumerate(person_boxes):
                    h_status = helmet_statuses[i]
                    h_conf   = helmet_confs[i]

                    if h_status == 1:   # WITHOUT HELMET
                        person_cx   = x1 + bw // 2
                        size_weight = (bw * bh) / (frame_w * frame_h)
                        dist        = abs(person_cx - frame_cx)
                        pos_weight  = 1 - (dist / frame_w)
                        score       = 2.0 * h_conf * size_weight * pos_weight
                        new_anomalies.append({
                            "score"  : score,
                            "cx"     : person_cx,
                            "box"    : (x1, y1, bw, bh),
                            "conf"   : h_conf,
                            "ratio_h": bh / frame_h,
                            "ratio_w": bw / frame_w,
                        })
                    elif h_status == 0:  # WITH HELMET
                        label = f'With Helmet {h_conf:.2f}'
                        new_with_helmet.append((x1, y1, bw, bh, label))

            # Sort anomalies
            new_anomalies.sort(key=lambda x: x["score"], reverse=True)
            new_anomalies = new_anomalies[:MAX_ANOMALIES]

            # Persist
            last_anomalies   = new_anomalies
            last_with_helmet = new_with_helmet
            last_vehicles    = new_vehicles

            # Update drone target
            if last_anomalies and last_anomalies[0]["score"] > SCORE_THRESHOLD:
                best = last_anomalies[0]
                state.update_target(best["cx"], best["ratio_h"], best["ratio_w"])
            else:
                state.clear_target()

            autonomous_control(frame_cx, frame_h, frame_w)

        # ── DRAW on every frame ────────────────────────────────────
        for (x1, y1, bw, bh, label) in last_with_helmet:
            draw_box(img, x1, y1, bw, bh, (0, 200, 0), label)

        for (x1, y1, bw, bh, label) in last_vehicles:
            draw_box(img, x1, y1, bw, bh, (255, 140, 0), label)

        for i, a in enumerate(last_anomalies):
            x1, y1, bw, bh = a["box"]
            color = (0, 0, 255) if i == 0 else (0, 80, 220)
            label = f"#{i+1} NO HELMET {a['score']:.2f}"
            draw_box(img, x1, y1, bw, bh, color, label)

        # ── HUD ────────────────────────────────────────────────────
        with state.lock:
            s     = state.status if state.autonomous else \
                    ("ARMED | MANUAL" if state.armed else "DISARMED | MANUAL")
            thr   = state.throttle
            yw    = state.yaw
            pt    = state.pitch
            rl    = state.roll
            auto  = state.autonomous
            armed = state.armed

        hud_c = (0, 220, 0) if armed else (60, 60, 200)
        cv2.rectangle(img, (0, 0), (450, 148), (0, 0, 0), -1)
        cv2.putText(img, f"STATUS : {s}",                (10, 22),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_c, 2, cv2.LINE_AA)
        cv2.putText(img, f"MODE   : {'AUTO' if auto else 'MANUAL'}", (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2, cv2.LINE_AA)
        cv2.putText(img, f"THR={thr}  YAW={yw}",         (10, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(img, f"PITCH={pt}  ROLL={rl}",        (10, 92),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        fps = 1.0 / max(time.time() - loop_start, 0.001)
        cv2.putText(img, f"FPS: {fps:.1f}",               (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (80,255,80),  1, cv2.LINE_AA)
        cv2.putText(img, "P=Auto  E=Arm  Q=Disarm",       (10, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140,140,140),1, cv2.LINE_AA)

        # ── DISPLAY — downscale for faster imshow ──────────────────
        if disp_scale < 1.0:
            disp_w = int(frame_w * disp_scale)
            disp_h = int(frame_h * disp_scale)
            display_img = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        else:
            display_img = img

        cv2.imshow("Drone Anomaly Controller", display_img)

        # ── FRAME RATE CAP — prevents flooding pipeline ────────────
        elapsed  = time.time() - loop_start
        wait_ms  = max(1, int((frame_delay - elapsed) * 1000))
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[!] Emergency stop triggered.")

finally:
    print("Sending disarm burst...")
    with state.lock:
        state.arm      = 1000
        state.throttle = THROTTLE_LOW
        state.pitch = state.roll = state.yaw = CENTER
    for _ in range(15):
        publish_rc()
        time.sleep(0.05)
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    cap.release()
    cv2.destroyAllWindows()
    print("Shutdown complete.")