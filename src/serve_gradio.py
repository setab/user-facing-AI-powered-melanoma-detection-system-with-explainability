import os
import json
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np

import gradio as gr
from torchcam.methods import GradCAM
from torchvision.transforms.functional import to_pil_image
from src.inference.xai import load_temperature, load_operating_points, apply_temperature
from src.config import Config


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_label_map(path: str) -> Dict[str, int]:
    with open(path, "r") as f:
        label_map = json.load(f)
    # Ensure consistent order
    return label_map


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(weights_path: str, label_map_path: str) -> Tuple[nn.Module, list]:
    labels_order = list(load_label_map(label_map_path).keys())
    model = build_model(num_classes=len(labels_order))
    state = torch.load(weights_path, map_location="cpu")
    # Accept either pure state_dict or checkpoint dict
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    elif isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model, labels_order


def get_transforms():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_and_explain(img: Image.Image, model: nn.Module, labels: list, temperature: Optional[float], op: Optional[dict]):
    tfm = get_transforms()
    x = tfm(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        if temperature is not None:
            logits = apply_temperature(logits, temperature)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))

    # GradCAM explanation
    cam_extractor = GradCAM(model, target_layer='layer4')
    # Forward again for CAM (needs gradients)
    logits = model(x)
    activation_maps = cam_extractor(class_idx=pred_idx, scores=logits)
    cam = activation_maps[0].squeeze().cpu()

    # Ensure 2D (H, W) for overlay
    if cam.ndim == 3:
        cam = cam.squeeze(0)
        if cam.ndim == 3:  # still 3D (C,H,W), take first channel
            cam = cam[0]

    # Resize CAM to image size and overlay
    img_resized = img.resize((224, 224))
    cam_img = to_pil_image(cam, mode='F').resize(img_resized.size)
    cam_arr = np.array(cam_img)
    cam_arr = (cam_arr - cam_arr.min()) / (cam_arr.ptp() + 1e-8)
    heatmap = (plt_colormap(cam_arr)[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.5 * np.array(img_resized) + 0.5 * heatmap).astype(np.uint8)

    result = Image.fromarray(overlay)

    # Clean up hooks
    try:
        cam_extractor.remove_hooks()
    except Exception:
        pass

    prob_dict = {label: float(probs[i]) for i, label in enumerate(labels)}

    # Melanoma verdict using operating point if available
    melanoma_decision = "N/A"
    if op is not None:
        mel_idx = int(op.get('class_index', -1))
        thr_key = 'melanoma_spec95'
        threshold = float(op.get('thresholds', {}).get(thr_key, 0.5))
        if mel_idx >= 0 and mel_idx < len(labels):
            mel_prob = float(probs[mel_idx])
            melanoma_decision = f"p={mel_prob:.3f} | thr={threshold:.3f} → {'melanoma' if mel_prob >= threshold else 'non-melanoma'}"

    return result, labels[pred_idx], prob_dict, melanoma_decision


def plt_colormap(arr: np.ndarray) -> np.ndarray:
    # Simple jet-like colormap without requiring matplotlib
    # Map [0,1] -> RGB using a few segments
    x = np.clip(arr, 0, 1)
    r = np.clip(1.5 - np.abs(2*x - 1.5), 0, 1)
    g = np.clip(1.5 - np.abs(2*x - 1.0), 0, 1)
    b = np.clip(1.5 - np.abs(2*x - 0.5), 0, 1)
    return np.stack([r, g, b, np.ones_like(r)], axis=-1)


def make_interface(model: nn.Module, labels: list, temperature: Optional[float], op: Optional[dict]):
    def _fn(image: Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        overlay, pred_label, prob_dict, decision = predict_and_explain(image, model, labels, temperature, op)
        # Sort probabilities descending
        items = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
        table = {k: round(v, 4) for k, v in items}
        return overlay, pred_label, table, decision

    return gr.Interface(
        fn=_fn,
        inputs=gr.Image(type="pil", label="Upload skin lesion image"),
        outputs=[
            gr.Image(type="pil", label="Grad-CAM Explanation"),
            gr.Label(label="Predicted Class"),
            gr.JSON(label="Class Probabilities"),
            gr.Textbox(label="Melanoma Decision (calibrated)")
        ],
        title="Melanoma Detection with XAI (Grad-CAM)",
        description="Upload an image. The model predicts the class and shows a heatmap highlighting regions that influenced the decision. Probabilities are temperature-calibrated; melanoma verdict uses an operating threshold (spec≈95%).",
        allow_flagging="never",
    )


def main():
    # Load config
    Config.validate()
    
    weights = str(Config.WEIGHTS_PATH)
    label_map_path = str(Config.LABEL_MAP_PATH)
    
    model, labels = load_model(weights, label_map_path)
    
    # Load calibration and operating points
    temperature = load_temperature(str(Config.TEMPERATURE_JSON_PATH))
    op = load_operating_points(str(Config.OPERATING_JSON_PATH))
    
    demo = make_interface(model, labels, temperature, op)
    
    # Launch with authentication if configured
    launch_kwargs = {
        'server_name': Config.GRADIO_SERVER_NAME,
        'server_port': Config.GRADIO_SERVER_PORT,
        'share': Config.GRADIO_SHARE,
    }
    
    if Config.has_auth():
        launch_kwargs['auth'] = Config.get_auth()
        print(f"🔒 Authentication enabled for user: {Config.GRADIO_USERNAME}")
    else:
        print("⚠️  WARNING: No authentication set. Anyone can access this interface.")
        print("   Set GRADIO_USERNAME and GRADIO_PASSWORD in .env for security.")
    
    print(f"🚀 Starting Gradio on {Config.GRADIO_SERVER_NAME}:{Config.GRADIO_SERVER_PORT}")
    
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
