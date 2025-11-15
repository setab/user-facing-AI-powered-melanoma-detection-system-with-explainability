import json
import os
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

from PIL import Image
import numpy as np

try:
    from torchcam.methods import GradCAM
    from torchcam.utils import overlay_mask
except Exception:
    GradCAM = None
    overlay_mask = None


def load_label_map(label_map_path: str) -> Dict[str, int]:
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    return label_map


def invert_label_map(label_map: Dict[str, int]):
    inv = {v: k for k, v in label_map.items()}
    return inv


def build_model(num_classes: int):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(weights_path: str, label_map_path: str, device: torch.device):
    label_map = load_label_map(label_map_path)
    model = build_model(num_classes=len(label_map))
    state = torch.load(weights_path, map_location=device)
    # Handle checkpoints saved as {'state_dict': ..., 'label_map': ...} or raw state_dict
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, label_map


def preprocess_image(image_path: str, img_size: int = 224):
    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pil_img = Image.open(image_path).convert('RGB')
    tensor = tf(pil_img).unsqueeze(0)
    return pil_img.resize((img_size, img_size)), tensor


@torch.no_grad()
def predict(model: torch.nn.Module, inp: torch.Tensor, device: torch.device, temperature: Optional[float] = None):
    logits = model(inp.to(device))
    if temperature is not None and temperature > 0:
        logits = apply_temperature(logits, temperature)
    probs = torch.softmax(logits, dim=1)
    return logits, probs


def gradcam_overlay(model: torch.nn.Module,
                    inp: torch.Tensor,
                    pil_img: Image.Image,
                    class_idx: int,
                    device: torch.device) -> Image.Image:
    if GradCAM is None or overlay_mask is None:
        raise RuntimeError("torchcam is not available; install it to use Grad-CAM.")
    cam_extractor = GradCAM(model, target_layer='layer4')
    inp = inp.to(device)
    scores = model(inp)
    cam = cam_extractor(class_idx, scores)[0].detach().cpu()
    # Normalize CAM
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    # Ensure 2D (H, W) for PIL conversion
    if cam.ndim == 3:
        cam = cam.squeeze(0)
        if cam.ndim == 3:  # still 3D (C,H,W), take first channel
            cam = cam[0]
    cam_img = T.ToPILImage()(cam)
    result = overlay_mask(pil_img, cam_img.resize(pil_img.size), alpha=0.5)
    # Clean up hooks
    try:
        cam_extractor.remove_hooks()
    except Exception:
        pass
    return result


# ---- Calibration & Operating Points Utilities ----
def load_temperature(path: str = "models/checkpoints/temperature.json") -> Optional[float]:
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        t = float(data.get('temperature', None))
        return t
    except Exception:
        return None


def load_operating_points(path: str = "models/checkpoints/operating_points.json") -> Optional[dict]:
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except Exception:
        return None


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    # safe guard against non-positive
    t = max(1e-6, float(temperature))
    return logits / t
