# visualize.py
import torch
from src.model import build_model
from src.gradcam import show_gradcam

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)

# Path to test image (NORMAL or PNEUMONIA)
image_path = "C:/automated pneumonia detection/chest_xray/test/NORMAL/IM-0011-0001.jpeg"

# Show Grad-CAM heatmap
show_gradcam(model, image_path, device)
