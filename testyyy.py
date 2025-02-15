import cv2
import matplotlib.pyplot as plt

import os

# Dossier contenant les images des produits
product_folder = r"D:\ESI\2CS\Algiers Up\AUP-2025\data\Photos produits Ramy"

# Charger l'image d'étagère
image_etagere = cv2.imread(r"D:\ESI\2CS\Algiers Up\AUP-2025\data\Photos rayonnages\pdv_pic_581631_1736000306767_98552.png", cv2.IMREAD_GRAYSCALE)

for produit in os.listdir(product_folder):
    product_path = os.path.join(product_folder, produit)

    image_produit = cv2.imread(product_path, cv2.IMREAD_GRAYSCALE)

    if image_produit is None:
        continue

    # ORB feature matching
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image_produit, None)
    kp2, des2 = orb.detectAndCompute(image_etagere, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) > 10:  # Si on trouve assez de correspondances
        print(f"Produit détecté : {produit}")
