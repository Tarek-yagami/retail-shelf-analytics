from ultralytics import YOLO
import torch


# Vérifier si un GPU est disponible
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Définir le fichier de configuration data.yaml
data_yaml = "data.yaml"  # Assurez-vous que ce fichier est dans le dossier racine du dataset


# Charger le modèle pré-entraîné YOLOv8
model = YOLO("yolov8n.pt")


# Entraîner le modèle avec les données téléchargées
model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=8,
    device=device
)


# Charger le meilleur modèle entraîné
model = YOLO("C:/Users/tarek/runs/detect/train5/weights/best.pt")


# Fonction pour tester le modèle sur une image
def detect_products(image_path):
    results = model(image_path, save=True, conf=0.5)
    return results


# Exemple d'utilisation sur une image de test
detect_products(r"D:\ESI\2CS\Algiers Up\Roboflow_test\AUP_merchandising.v2i.yolov8\valid\images\WhatsApp-Image-2025-02-05-a-12_17_26_f27f3f0b_jpg.rf.aeb030e08a547283938fef054f7ae657.jpg")
