from pathlib import Path
import torch

# Paths
DATA_DIR = Path("data_processed")
MODEL_DIR = Path("models")
MODEL_FILENAME = "plant_disease_resnet18.pth"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

# Training config
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
IMG_SIZE = 224

# Device config
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]
