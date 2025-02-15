from ultralytics import YOLO

# Charger un modèle pré-entraîné YOLOv8
model = YOLO("yolov8n.pt")  # Utilisation d’un modèle de base

# Entraîner sur notre dataset
model.train(data="data.yaml", epochs=10, imgsz=640, batch=8, device="cpu")  
