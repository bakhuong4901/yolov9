from ultralytics import YOLO
model = YOLO("runs/segment/train/weights/best.pt")
results = model.predict(source="val_avai/Subset_S2B_MSIL2A_20220726T030529_N0400_R075_T49PBR_20220726T062302_RGB.png",save = True,conf=0.5,  # Ngưỡng tin cậy cao hơn để lọc các dự đoán không chắc chắn
    iou=0.4,   # Ngưỡng IoU cho Non-Maximum Suppression
    max_det=10)