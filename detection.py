import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Charger le modèle entraîné
model = YOLO(" C:/Users/tarek/runs/detect/train5/weights/best.pt")  # Vérifie le chemin exact du modèle sauvegardé

# Charger une image de rayonnage
image_path = r"D:\ESI\2CS\Algiers Up\AUP-2025\data\Photos rayonnages\pdv_pic_571111_1736332765032_98552.png"
image = cv2.imread(image_path)

# Effectuer la détection
results = model(image_path)

# Afficher les résultats
for result in results:
    result.show()  # Affichage interactif
