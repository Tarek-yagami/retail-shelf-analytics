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
model = YOLO("C:/Users/tarek/runs/detect/train11/weights/best.pt")


# Fonction pour tester le modèle sur une image
def detect_products(image_path):
    results = model(image_path)  # Effectuer la détection sur l'image
    
    # Dictionnaire pour compter le nombre d'occurrences par catégorie
    product_counts = {}

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())  # Récupérer l'ID de la classe
            class_name = model.names[class_id]  # Récupérer le nom de la classe
            
            # Incrémenter le compteur de la catégorie
            product_counts[class_name] = product_counts.get(class_name, 0) + 1

    return product_counts  # Retourner un dictionnaire avec le nombre de produits détectés par catégorie

# Exemple d'utilisation sur une image de test
resultats = detect_products(r"D:\ESI\2CS\Algiers Up\AUP-2025\AUP_merchandising.v2i.yolov8\valid\images\WhatsApp-Image-2025-02-08-a-17_21_10_de969159_jpg.rf.a13e65551ebf0807ae37e28b081441f4.jpg")
print(resultats)  # Afficher le nombre de produits par catégorie


