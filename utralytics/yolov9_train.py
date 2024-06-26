from ultralytics import YOLO


# Build a YOLOv9c model from pretrained weight
model = YOLO("yolov9c-seg.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="/Users/nguyenbakhuong/Downloads/yolov9/data/data.yaml", epochs=300, imgsz=640,batch=16)

# Run inference with the YOLOv9c model on the 'bus.jpg' image
# results = model("path/to/bus.jpg")