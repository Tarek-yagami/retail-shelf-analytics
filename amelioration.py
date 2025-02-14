from PIL import Image, ImageEnhance
import os
import random

input_folder = "Photos produit Ramy/"
output_folder = "Dataset/images/train/"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        img = Image.open(os.path.join(input_folder, filename))

        # Appliquer des transformations
        for i in range(5):  # Générer 5 variations par image
            img_transformed = img.rotate(random.randint(-15, 15))  # Rotation
            enhancer = ImageEnhance.Brightness(img_transformed)
            img_transformed = enhancer.enhance(random.uniform(0.8, 1.2))  # Changer luminosité
            
            img_transformed.save(os.path.join(output_folder, f"{filename[:-4]}_{i}.png"))
