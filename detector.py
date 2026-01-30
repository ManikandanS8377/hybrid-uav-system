import cv2
import math
import time
import cvzone
from ultralytics import YOLO

# =========================================================
# PLATFORM CONFIG (WINDOWS MODE)
# =========================================================
IS_PI = False   # Windows → mock servo

def set_servo_angle(angle):
    print(f"[MOCK SERVO] Angle -> {int(angle)}°")

# =========================================================
# SERVO LOGIC PARAMETERS
# =========================================================
HOME_ANGLE = 90
servo_angle = HOME_ANGLE

Kp = 0.002
MAX_STEP = 3
CENTER_DEADZONE = 25
SCORE_THRESHOLD = 0.02
ALPHA = 0.85

prev_error = 0

ANOMALY_HOLD_TIME = 5  # seconds
last_anomaly_time = 0
anomaly_active = False

# =========================================================
# VIDEO & YOLO SETUP
# =========================================================
cap = cv2.VideoCapture("Media/traffic.mp4")

coco_model = YOLO("yolov8n.pt")
helmet_model = YOLO("Weights/best.pt")

helmet_classNames = ['With Helmet', 'Without Helmet']

# COCO class names for vehicles
coco_classNames = {
    0: 'Person',
    2: 'Car',
    3: 'Motorcycle',
    5: 'Bus',
    7: 'Truck'
}

# Performance optimization
frame_skip = 3  # Process every 3rd frame
frame_count = 0

# Maximum anomalies to display
max_anomalies = 5

frame_w, frame_h = None, None

# =========================================================
# MAIN LOOP
# =========================================================
while True:
    success, img = cap.read()
    if not success:
        break
    
    frame_count += 1

    if frame_w is None:
        frame_h, frame_w = img.shape[:2]
        frame_cx = frame_w // 2

    # Skip frames for performance
    if frame_count % frame_skip != 0:
        cv2.imshow("Helmet Anomaly Tracking (Windows Mock Servo)", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    anomalies = []

    coco_results = coco_model(img, stream=True)

    for r in coco_results:
        for box in r.boxes:
            cls = int(box.cls[0])

            if cls == 0:  # PERSON
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                if w <= 0 or h <= 0:
                    continue

                person_crop = img[y1:y2, x1:x2]
                helmet_results = helmet_model(person_crop, stream=True)

                helmet_status = None
                helmet_conf = 0

                for h_r in helmet_results:
                    if len(h_r.boxes) > 0:
                        hb = h_r.boxes[0]
                        helmet_status = int(hb.cls[0])
                        helmet_conf = float(hb.conf[0])
                        break

                if helmet_status == 1:  # WITHOUT HELMET (ANOMALY)
                    person_cx = x1 + w // 2

                    size_weight = (w * h) / (frame_w * frame_h)
                    distance = abs(person_cx - frame_cx)
                    position_weight = 1 - (distance / frame_w)
                    violation_weight = 2.0

                    score = violation_weight * helmet_conf * size_weight * position_weight

                    anomalies.append({
                        "score": score,
                        "cx": person_cx,
                        "box": (x1, y1, w, h),
                        "conf": helmet_conf
                    })
                else:
                    # Draw normal person with helmet
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=(0, 255, 0))
                    if helmet_status is not None:
                        helmet_label = f'{helmet_classNames[helmet_status]} {helmet_conf:.2f}'
                        cvzone.putTextRect(img, helmet_label, (x1, max(30, y1 - 10)), colorR=(0, 255, 0))
            
            # Detect vehicles (motorcycles, cars, trucks, buses)
            elif cls in [2, 3, 5, 7]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=(255, 0, 0))
                
                conf = float(box.conf[0])
                vehicle_label = f'{coco_classNames[cls]} {conf:.2f}'
                cvzone.putTextRect(img, vehicle_label, (x1, max(30, y1 - 10)), colorR=(255, 0, 0))
    
    # Sort anomalies by score (highest first) and limit to max_anomalies
    anomalies.sort(key=lambda x: x['score'], reverse=True)
    anomalies = anomalies[:max_anomalies]
    
    # Draw only the top anomalies with red boxes
    for anomaly in anomalies:
        x1, y1, w, h = anomaly['box']
        cvzone.cornerRect(img, (x1, y1, w, h), colorC=(0, 0, 255))
        cvzone.putTextRect(
            img,
            f"NO HELMET {anomaly['score']:.2f}",
            (x1, max(30, y1 - 10)),
            colorR=(0, 0, 255)
        )

    # =====================================================
    # SERVO DECISION (HIGHEST VALUE ANOMALY)
    # =====================================================
    current_time = time.time()

    if len(anomalies) > 0:
        target = max(anomalies, key=lambda x: x["score"])

        if target["score"] > SCORE_THRESHOLD:
            anomaly_active = True
            last_anomaly_time = current_time

            error = target["cx"] - frame_cx

            if abs(error) < CENTER_DEADZONE:
                error = 0

            smoothed_error = ALPHA * prev_error + (1 - ALPHA) * error
            prev_error = smoothed_error

            delta = Kp * smoothed_error
            delta = max(-MAX_STEP, min(MAX_STEP, delta))

            servo_angle += delta
            servo_angle = max(0, min(180, servo_angle))

            set_servo_angle(servo_angle)

    # =====================================================
    # RETURN TO NORMAL AFTER 5 SECONDS
    # =====================================================
    if anomaly_active:
        if current_time - last_anomaly_time > ANOMALY_HOLD_TIME:
            servo_angle = HOME_ANGLE
            set_servo_angle(servo_angle)
            anomaly_active = False

    # =====================================================
    # DISPLAY
    # =====================================================
    cv2.putText(
        img,
        f"Servo Angle: {int(servo_angle)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Helmet Anomaly Tracking (Windows Mock Servo)", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
