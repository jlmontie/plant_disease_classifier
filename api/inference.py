from PIL import Image
import torch
from torchvision import models, transforms
from plant_disease_classifier.config import DEVICE, IMG_SIZE, MODEL_PATH, CLASS_NAMES

# Define image transforms (must match training)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict_image(model, image: Image.Image):
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probabilities, 1)
        return {
            "class": CLASS_NAMES[pred.item()],
            "confidence": round(conf.item() * 100, 2)
        }