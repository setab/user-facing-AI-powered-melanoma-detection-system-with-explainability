import json
from typing import Tuple, Dict

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
    model.load_state_dict(state)
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
def predict(model: torch.nn.Module, inp: torch.Tensor, device: torch.device):
    logits = model(inp.to(device))
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
    cam_img = T.ToPILImage()(cam.unsqueeze(0))
    result = overlay_mask(pil_img, cam_img, alpha=0.5)
    return result
