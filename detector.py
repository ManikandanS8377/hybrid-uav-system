import cv2
import math
import cvzone
from ultralytics import YOLO

# Initialize video capture
video_path = "Media/traffic.mp4"
cap = cv2.VideoCapture(video_path)

# Load YOLO models
coco_model = YOLO("yolov8n.pt")  # Built-in COCO model for person and vehicle detection
helmet_model = YOLO("Weights/best.pt")  # Custom helmet detection model

# Define COCO class names (relevant ones)
coco_classNames = {
    0: 'Person',
    2: 'Car',
    3: 'Motorcycle',
    5: 'Bus',
    7: 'Truck'
}

# Define helmet class names
helmet_classNames = ['With Helmet', 'Without Helmet']

# For the use of Webcam
# Open the webcam (use 0 for the default camera, or 1, 2, etc. for additional cameras)
# cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    
    # Run COCO model for person and vehicle detection
    coco_results = coco_model(img, stream=True)
    
    person_detected = False
    
    for r in coco_results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            
            # Check if person is detected
            if cls == 0:  # Person class
                person_detected = True
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Crop the person region for helmet detection
                person_crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
                
                # Run custom helmet detection on the person
                helmet_results = helmet_model(person_crop, stream=True)
                helmet_status = None
                helmet_conf = 0
                
                for h_r in helmet_results:
                    h_boxes = h_r.boxes
                    if len(h_boxes) > 0:
                        h_box = h_boxes[0]  # Take the first detection
                        helmet_status = int(h_box.cls[0])
                        helmet_conf = math.ceil((h_box.conf[0] * 100)) / 100
                        break
                
                # Draw bounding box and helmet status
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=(0, 255, 0))
                
                if helmet_status is not None:
                    helmet_label = f'{helmet_classNames[helmet_status]} {helmet_conf}'
                    cvzone.putTextRect(img, helmet_label, (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0, 255, 0))
                else:
                    cvzone.putTextRect(img, 'Person Detected', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(255, 255, 0))
            
            # Detect vehicles (motorcycles, cars, trucks, buses)
            elif cls in [2, 3, 5, 7]:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=(255, 0, 0))
                
                conf = math.ceil((box.conf[0] * 100)) / 100
                vehicle_label = f'{coco_classNames[cls]} {conf}'
                cvzone.putTextRect(img, vehicle_label, (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(255, 0, 0))
    
    # Display status
    status_text = "Person with Helmet Status: Running" if person_detected else "No Person Detected - Vehicle Detection Only"
    cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Hybrid Helmet & Vehicle Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()