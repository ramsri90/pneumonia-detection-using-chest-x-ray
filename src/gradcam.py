import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        target_layer.register_forward_hook(forward_hook)
        if hasattr(target_layer, "register_full_backward_hook"):
            target_layer.register_full_backward_hook(backward_hook)
        else:
            target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        loss = output[0, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224), dtype=np.float32)

        gradients = self.gradients[0].cpu().detach().numpy()
        activations = self.activations[0].cpu().detach().numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        range_val = np.max(cam) - np.min(cam)
        if range_val > 0:
            cam = (cam - np.min(cam)) / range_val
        else:
            cam = np.zeros_like(cam)
        return cam


def compute_gradcam_blend(model, image: Image.Image, device=None, class_idx=None, alpha: float = 0.45) -> tuple[Image.Image, Image.Image]:
    """
    Computes Grad-CAM and returns:
    1. Blended PIL Image at requested alpha (0.0 = original, 1.0 = pure heatmap)
    2. Pure colormap heatmap PIL Image
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_rgb = image.convert("RGB")
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    target_layer = model.layer4[1].conv2
    cam = GradCAM(model, target_layer)
    heatmap = cam.generate_heatmap(input_tensor, class_idx=class_idx)

    img_np = np.array(img_rgb.resize((224, 224)))
    heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

    alpha_clamped = max(0.0, min(1.0, float(alpha)))
    superimposed = cv2.addWeighted(img_np, 1.0 - alpha_clamped, heatmap_rgb, alpha_clamped, 0)

    return Image.fromarray(superimposed), Image.fromarray(heatmap_rgb)


def compute_gradcam_overlay(model, image: Image.Image, device=None, class_idx=None) -> Image.Image:
    blended, _ = compute_gradcam_blend(model, image, device=device, class_idx=class_idx, alpha=0.45)
    return blended


# Utility to visualize result via CLI / Matplotlib
def show_gradcam(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    superimposed_pil = compute_gradcam_overlay(model, image, device)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(superimposed_pil)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
