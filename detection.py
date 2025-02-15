import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import yaml
import os
# Load class names from the YAML file
def load_class_names(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return data["names"]  # Returns a dictionary {0: "Bottle", 1: "Can", 2: "Box"}

# Load class names from your dataset
class_names = load_class_names(r"D:\ESI\2CS\Algiers Up\AUP-2025\AUP_merchandising.v2i.yolov8\data.yaml")

# Charger le modèle entraîné
model = YOLO(" D:/ESI/2CS/Algiers Up/AUP-2025/detection/train13/weights/best.pt")  # Vérifie le chemin exact du modèle sauvegardé

# Fonction pour tester le modèle sur une image et enregistrer les résultats
def detect_products(image_path):
    results = model(image_path, save=True, project=os.getcwd(), name="predictions", conf=0.6)
# Dictionary to count occurrences of each class
    object_counts = {}

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0].item())  # Extract class ID
            class_name = class_names[class_id]  # Map class ID to name

            # Update object count
            object_counts[class_name] = object_counts.get(class_name, 0) + 1

    print("\nDetected Objects Summary:")
    for cls, count in object_counts.items():
        print(f" - {cls}: {count}")

    return object_counts

# Exemple d'utilisation sur une image de test
test_image = r"D:\ESI\2CS\Algiers Up\AUP-2025\AUP_merchandising.v2i.yolov8\valid\images\WhatsApp-Image-2025-02-11-a-13_41_24_74bfd9d4_jpg.rf.ce1a51db7c628276841ebe0b3e20a7cb.jpg"
detect_products(test_image)