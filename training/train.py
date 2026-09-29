from ultralytics import YOLO


MODEL = "yolo11n.pt"

model = YOLO(MODEL)

results = model.train(
    data="training/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    patience=15,
    workers=2,
    project="training/runs",
    name="snapinspect"
)

print("Training completed.")

print(
    "Best model:"
    " training/runs/snapinspect/weights/best.pt"
)
