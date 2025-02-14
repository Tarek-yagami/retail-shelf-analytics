from ultralytics import YOLO

# Charger un modèle pré-entraîné YOLOv8
model = YOLO("yolov8n.pt")  # Utilisation d’un modèle de base

# Entraîner sur notre dataset
model.train(data="data.yaml", epochs=50, imgsz=640, batch=8, device="cuda")  # Utiliser GPU si disponible
