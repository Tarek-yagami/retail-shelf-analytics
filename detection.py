import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Charger le modèle entraîné
model = YOLO("runs/train/exp/weights/best.pt")  # Vérifie le chemin exact du modèle sauvegardé

# Charger une image de rayonnage
image_path = "Photos rayonnages/image_test.jpg"
image = cv2.imread(image_path)

# Effectuer la détection
results = model(image_path)

# Afficher les résultats
for result in results:
    result.show()  # Affichage interactif
