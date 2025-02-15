import os
import shutil
from sklearn.model_selection import train_test_split

dataset_dir = 'D:\\ESI\\2CS\\Algiers Up\\AUP-2025\\data\\Photos rayonnages'
train_dir = 'D:\\ESI\\2CS\\Algiers Up\\AUP-2025\\data\\Photos rayonnages\\train'
val_dir = 'D:\\ESI\\2CS\\Algiers Up\\AUP-2025\\data\\Photos rayonnages\\val'

# Get a list of all files in the dataset directory
all_files = os.listdir(dataset_dir)

# Split the files into training and validation sets (e.g., 80% training, 20% validation)
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

# Create train and validation directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Move files to their respective directories
for file in train_files:
    shutil.move(os.path.join(dataset_dir, file), os.path.join(train_dir, file))

for file in val_files:
    shutil.move(os.path.join(dataset_dir, file), os.path.join(val_dir, file))

print("Dataset has been split into train and validation sets.")
