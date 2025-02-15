from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import torch
import os
from PIL import Image

devices = 'cuda' if torch.cuda.is_available() else 'cpu'

# Charger le modèle YOLO entraîné
MODEL_PATH = "D:/ESI/2CS/Algiers Up/AUP-2025/detection/train13/weights/best.pt"
model = YOLO(MODEL_PATH)

# Liste des catégories
CLASSES = ['7up_plastique', 'arwa_canette', 'arwa_plastique', 'belgaid_plastique', 'bifa_carton', 'candia_carton',
           'el_arabi_plastique', 'fanta_plastique', 'hamoud_plastique', 'ifri_plastique', 'ifruit', 'ifruit_carton',
           'ifruit_plastique', 'metidja_plastique', 'miranda_plastique', 'ngaous_carton', 'ngaous_plastique',
           'obei_carton', 'orangina_plastique', 'pepsi_plastique', 'ramy_canette', 'ramy_carton', 'ramy_plastique',
           'rani_plastique', 'rouiba_carton', 'rouiba_plastique', 'skander_plastique', 'sommam_carton',
           'sprite_canette', 'sprite_plastique', 'tassilo_carton', 'tastetea_plastique', 'tazej_plastique',
           'tchina_plastique', 'toudja_carton']

# Initialiser l’application Flask
app = Flask(__name__)
@app.route('/')
def home():
    return send_file('test.html')
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image envoyée'}), 400
    
    image_file = request.files['image']
    image = Image.open(image_file)
    
    # Exécuter la détection
    results = model(image,save=True, project=os.getcwd(), name="predictions", conf=0.5)
    
    # Analyser les résultats
    detections = results[0].boxes.cls.cpu().numpy()  # Obtenir les classes détectées
    product_counts = {category: 0 for category in CLASSES}  # Initialiser le compteur
    
    for det in detections:
        class_name = CLASSES[int(det)]  # Récupérer le nom de la classe
        product_counts[class_name] += 1  # Incrémenter le compteur
    
    return jsonify({'products_detected': product_counts})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
