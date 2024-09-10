from ultralytics import YOLO


# Build a YOLOv9c model from scratch
model = YOLO("YOLOv9t.yaml")





# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data=r"C:\Users\harvi\Weld\wallet.v1i.yolov9\data.yaml", epochs=1, imgsz=640)



