import os
import shutil
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import torch
from torch.utils.data import Dataset
import random
from sklearn.model_selection import train_test_split
import unicodedata
from itertools import chain


def setup_directory_structure(base_dir="data"):
    """Creates the necessary directory structure including augmentation directories"""
    dirs = [
        f"{base_dir}/images/train",
        f"{base_dir}/images/val",
        f"{base_dir}/labels/train",
        f"{base_dir}/labels/val",
        f"{base_dir}/products_reference",
        f"{base_dir}/augmented/products",  # New directory for augmented products
        f"{base_dir}/augmented/shelves"    # New directory for augmented shelves
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return base_dir

def copy_product_images(products_dir, output_dir):
    """Copie les images de référence des produits"""
    products_dir = Path(products_dir)
    output_dir = Path(output_dir)
    for img_path in products_dir.glob("*.jpg"):
        shutil.copy(img_path, output_dir / "products_reference")

def split_shelf_images(base_dir, test_size=0.2):
    """Divides augmented shelf images into training and validation sets."""
    shelves_dir = Path(base_dir) / "augmented/shelves"  # Use augmented shelves directory
    base_dir = Path(base_dir)
    
    all_images = list(
        chain(
            shelves_dir.glob("*.jpg"),
            shelves_dir.glob("*.png")
        )
    )
    train_imgs, val_imgs = train_test_split(all_images, test_size=test_size, random_state=42)
    
    # Copy images to respective directories
    for img in train_imgs:
        shutil.copy(img, base_dir / "images/train")
    
    for img in val_imgs:
        shutil.copy(img, base_dir / "images/val")

def create_augmentation_pipeline():
    """Crée un pipeline d'augmentation pour les images d'étagères"""
    transform = A.Compose([
        # Transformations de base
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        
        # Transformations géométriques légères (pour ne pas déformer les produits)
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
    ])
    return transform

def safe_read_image(image_path):
    """Lit une image en gérant les caractères spéciaux dans le chemin"""
    image_path = Path(image_path)
    
    # Print the absolute path for debugging
    print(f"Trying to read image: {image_path.absolute()}")
    
    # Check if the file exists
    if not image_path.exists():
        print(f"File does not exist: {image_path.absolute()}")
        return None
    
    # First try - direct read
    img = cv2.imread(str(image_path))
    
    # If reading fails, try with Unicode normalization
    if img is None:
        try:
            normalized_path = unicodedata.normalize('NFC', str(image_path))
            print(f"Trying normalized path: {normalized_path}")
            img = cv2.imread(normalized_path)
            
            if img is None:
                print(f"ERROR: Unable to load image: {image_path}")
                print(f"Check that the file exists and the path is correct.")
                print(f"Absolute path: {image_path.absolute()}")
                print(f"File exists: {image_path.exists()}")
        except Exception as e:
            print(f"Exception when reading image {image_path}: {e}")
    
    return img

def rename_files_in_directory(directory):
    """Renames files in the directory to remove special characters"""
    directory = Path(directory)
    for img_path in directory.glob("*"):
        new_name = unicodedata.normalize('NFKD', img_path.name).encode('ascii', 'ignore').decode('ascii')
        new_path = img_path.with_name(new_name)
        img_path.rename(new_path)
        print(f"Renamed {img_path} to {new_path}")

def generate_synthetic_annotations(shelf_image_path, product_reference_dir, output_label_path):
    """
    Génère des annotations synthétiques en plaçant aléatoirement des produits de référence
    sur l'image d'étagère (simulation simplifiée)
    """
    # Charger l'image d'étagère avec gestion des caractères spéciaux
    shelf_img = safe_read_image(shelf_image_path)
    
    if shelf_img is None:
        return  # Arrêter si l'image n'a pas pu être chargée
    
    h, w = shelf_img.shape[:2]
    
    # Charger les images de produits de référence
    product_reference_dir = Path(product_reference_dir)
    product_images = list(
        chain(
            product_reference_dir.glob("*.jpg"),
            product_reference_dir.glob("*.png")
        )
    )    
    annotations = []
    
    # Simuler 3 à 8 produits Ramy par étagère
    num_products = random.randint(3, 8)
    
    for _ in range(num_products):
        # Choisir un produit aléatoire
        product_path = random.choice(product_images)
        
        # Simuler une position aléatoire (simplifié)
        x1 = random.randint(0, w - 100)
        y1 = random.randint(0, h - 100)
        
        # Taille aléatoire du produit (simplifié)
        prod_width = random.randint(50, 100)
        prod_height = random.randint(100, 200)
        
        x2 = min(x1 + prod_width, w)
        y2 = min(y1 + prod_height, h)
        
        # Ajouter l'annotation
        annotations.append({
            "class_id": 0,  # 0 pour produit Ramy
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })
    
    # Écrire les annotations au format YOLO
    create_yolo_annotations(shelf_image_path, annotations, output_label_path)

def create_yolo_annotations(image_path, detections, output_path):
    """Crée des annotations au format YOLO"""
    img = safe_read_image(image_path)
    
    if img is None:
        return  # Arrêter si l'image n'a pas pu être chargée
    
    height, width = img.shape[:2]
    
    # Assurer que output_path est un objet Path
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        for det in detections:
            # Convertir les coordonnées en format YOLO (x_center, y_center, width, height)
            x_center = (det['x1'] + det['x2']) / (2 * width)
            y_center = (det['y1'] + det['y2']) / (2 * height)
            w = (det['x2'] - det['x1']) / width
            h = (det['y2'] - det['y1']) / height
            
            # Écrire l'annotation (class_id x_center y_center width height)
            f.write(f"{det['class_id']} {x_center} {y_center} {w} {h}\n")

def copy_and_augment_product_images(products_dir, base_dir, num_augmentations=5):
    """Copies and augments product images to a dedicated augmentation directory"""
    products_dir = Path(products_dir)
    base_dir = Path(base_dir)
    reference_dir = base_dir / "products_reference"
    augmented_dir = base_dir / "augmented/products"
    
    # Create directories if they don't exist
    os.makedirs(reference_dir, exist_ok=True)
    os.makedirs(augmented_dir, exist_ok=True)
    
    # Create augmentation pipeline for products
    product_transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=0.5),
            A.GaussianBlur(blur_limit=7, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            A.RandomShadow(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ], p=0.3),
    ])

    # Process each product image
    for img_path in chain(products_dir.glob("*.jpg"), products_dir.glob("*.png")):
        # Copy original to reference directory
        shutil.copy(img_path, reference_dir)
        
        # Load image for augmentation
        img = safe_read_image(img_path)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Generate augmented versions
        for aug_idx in range(num_augmentations):
            augmented = product_transform(image=img_rgb)
            aug_image = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
            
            # Save augmented image to augmented products directory
            aug_img_path = augmented_dir / f"{img_path.stem}_aug_{aug_idx}.jpg"
            cv2.imwrite(str(aug_img_path), aug_image)

def augment_shelf_images(base_dir, num_augmentations=5):
    """Augments shelf images and saves them to a dedicated directory"""
    base_dir = Path(base_dir)
    img_dir = base_dir / "Photos rayonnages"
    augmented_dir = base_dir / "augmented/shelves"
    
    # Create augmentation pipeline for shelves
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
    ])
    
    # Process each shelf image
    for img_path in img_dir.glob("*.jpg"):
        img = safe_read_image(img_path)
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Generate augmented versions
        for aug_idx in range(num_augmentations):
            augmented = transform(image=img_rgb)
            aug_image = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
            
            # Save augmented image to augmented shelves directory
            aug_img_path = augmented_dir / f"{img_path.stem}_aug_{aug_idx}.jpg"
            cv2.imwrite(str(aug_img_path), aug_image)

            
def create_data_yaml(base_dir, output_path="data.yaml"):
    """Crée le fichier YAML pour l'entraînement YOLOv5"""
    base_dir = Path(base_dir)
    content = f"""
train: {(base_dir / 'images/train').absolute()}
val: {(base_dir / 'images/val').absolute()}
nc: 1
names: ['ramy_product']
    """
    
    with open(output_path, 'w') as f:
        f.write(content)
    
    return output_path

def main(products_dir, shelves_dir):
    # Setup directory structure
    base_dir = setup_directory_structure()
    
    # Rename files to remove special characters
    rename_files_in_directory(shelves_dir)
    
    # Copy and augment product images to dedicated directory
    copy_and_augment_product_images(products_dir, base_dir)
    
    
    # Augment shelf images and save to dedicated directory
    augment_shelf_images(base_dir)
    
    # Split shelf images into train/val sets
    split_shelf_images(base_dir)
    
    # Generate synthetic annotations for training set
    base_dir = Path(base_dir)
    train_img_dir = base_dir / "images/train"
    train_label_dir = base_dir / "labels/train"
    product_ref_dir = base_dir / "products_reference"
    
    for img_path in train_img_dir.glob("*.jpg"):
        label_path = train_label_dir / f"{img_path.stem}.txt"
        generate_synthetic_annotations(img_path, product_ref_dir, label_path)
    
    # Generate synthetic annotations for validation set
    val_img_dir = base_dir / "images/val"
    val_label_dir = base_dir / "labels/val"
    
    for img_path in val_img_dir.glob("*.jpg"):
        label_path = val_label_dir / f"{img_path.stem}.txt"
        generate_synthetic_annotations(img_path, product_ref_dir, label_path)
    
    # Create YAML file for training
    yaml_path = create_data_yaml(base_dir)
    
    print(f"Data preparation completed. Directory structure:")
    print(f"- Original products: {base_dir}/products_reference")
    print(f"- Augmented products: {base_dir}/augmented/products")
    print(f"- Augmented shelves: {base_dir}/augmented/shelves")
    print(f"- Training images: {base_dir}/images/train")
    print(f"- Validation images: {base_dir}/images/val")
    print(f"YAML configuration file created: {yaml_path}")

if __name__ == "__main__":
    products_dir = "data/Photos produits Ramy"
    shelves_dir = "data/Photos rayonnages"
    main(products_dir, shelves_dir)