
from ultralytics import YOLO

model = YOLO("YOLO Prediction/Weights/best.pt")

metrics = model.val(
    data="Custom Prediction Model/data.yaml",
    split="test",
    imgsz=640
)

print(metrics)