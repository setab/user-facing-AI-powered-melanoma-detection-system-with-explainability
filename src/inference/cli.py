import argparse
import json
import sys

import torch

from src.inference.xai import (
    load_model,
    preprocess_image,
    predict,
    gradcam_overlay,
    invert_label_map,
    load_temperature,
    load_operating_points,
)


def ask_followup(base_prob: float) -> float:
    print("Confidence is low. Answer a couple of quick questions to refine the estimate.")
    # Basic example; extend with more nuanced questions/features
    q1 = input("Has the lesion changed in size/color recently? (y/n): ").strip().lower()
    q2 = input("Is the diameter larger than 6mm (pencil eraser)? (y/n): ").strip().lower()

    # Simple logistic-style update: bump probability with risk answers
    adj = base_prob
    if q1 == 'y':
        adj = min(1.0, adj + 0.10)
    if q2 == 'y':
        adj = min(1.0, adj + 0.10)
    return adj


def main():
    ap = argparse.ArgumentParser(description='Predict melanoma and save Grad-CAM overlay')
    ap.add_argument('--image', required=True, help='Path to input image (JPG/PNG)')
    ap.add_argument('--weights', default='melanoma_resnet50.pth')
    ap.add_argument('--label-map', default='label_map.json')
    ap.add_argument('--out', default='gradcam_overlay.jpg', help='Where to save Grad-CAM overlay image')
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--ask-threshold', type=float, default=0.6, help='If melanoma prob < threshold, ask follow-up questions')
    ap.add_argument('--temperature-json', default='models/checkpoints/temperature.json', help='Path to temperature.json')
    ap.add_argument('--operating-json', default='models/checkpoints/operating_points.json', help='Path to operating_points.json')
    ap.add_argument('--operating-key', default='melanoma_spec95', help='Operating point key to use from operating_points.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, label_map = load_model(args.weights, args.label_map, device)
    inv_map = invert_label_map(label_map)

    # Load calibration and operating points
    temperature = load_temperature(args.temperature_json)
    op = load_operating_points(args.operating_json)
    mel_idx = None
    threshold = None
    if op is not None:
        mel_idx = int(op.get('class_index', label_map.get('melanoma', -1)))
        threshold = float(op.get('thresholds', {}).get(args.operating_key, args.ask_threshold))
    else:
        mel_idx = label_map.get('melanoma', -1)
        threshold = args.ask_threshold

    pil_img, inp = preprocess_image(args.image, args.img_size)
    logits, probs = predict(model, inp, device, temperature=temperature)
    probs = probs.squeeze(0).cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_cls = inv_map[pred_idx]
    pred_prob = float(probs[pred_idx])

    print(f"Prediction: {pred_cls} | p={pred_prob:.3f}")
    if mel_idx is not None and mel_idx >= 0:
        mel_prob = float(probs[mel_idx])
        verdict = 'melanoma' if mel_prob >= threshold else 'non-melanoma'
        print(f"Calibrated melanoma p={mel_prob:.3f} | threshold={threshold:.3f} → VERDICT: {verdict}")

    # Save Grad-CAM overlay
    try:
        overlay = gradcam_overlay(model, inp, pil_img, pred_idx, device)
        overlay.save(args.out)
        print(f"Grad-CAM overlay saved to {args.out}")
    except Exception as e:
        print(f"Grad-CAM generation skipped: {e}")

    # Ask follow-up if low confidence and melanoma is a class
    # Optional: ask follow-up if melanoma probability is below threshold
    if mel_idx is not None and mel_idx >= 0:
        mel_prob = float(probs[mel_idx])
        if mel_prob < threshold:
            refined = ask_followup(mel_prob)
            print(f"Refined melanoma probability after Q&A: {refined:.3f}")


if __name__ == '__main__':
    main()
