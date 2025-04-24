"""
Predict Muffin vs Chihuahua
usage:
    python src/predict.py --img path/to/file.jpg \
                          --weights assets/muffin_vs_chihuahua.pth

Author : <AlimXiaoWei>
"""

import argparse, yaml, torch
from pathlib import Path
from PIL import Image
from torchvision import transforms, models

# _______________ CLI _______________
parser = argparse.ArgumentParser()
parser.add_argument("--img",     required=True,  help="Path to image")
parser.add_argument("--weights", default="assets/muffin_vs_chihuahua.pth")
parser.add_argument("--config",  default="config/config.yaml")
args = parser.parse_args()

# _______________ Conf & Classes _______________
with open(args.config) as f:
    cfg = yaml.safe_load(f)

class_names = ["chihuahua", "muffin"]

# _______________ Model _______________
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(args.weights, map_location=device))
model.to(device).eval()

# _______________ Image Transformation _______________
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

img = Image.open(args.img).convert("RGB")
tensor = transform(img).unsqueeze(0).to(device)

# _______________ Prediction _______________
with torch.no_grad():
    logits = model(tensor)
    pred   = torch.argmax(logits, 1).item()
print(f"Prediction: {class_names[pred].upper()} ✅")
