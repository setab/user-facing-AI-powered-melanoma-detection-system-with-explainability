import os
import unittest
from pathlib import Path

import torch

from src.inference.xai import (
    load_model,
    preprocess_image,
    predict,
    load_temperature,
    load_operating_points,
)


class SmokeInferenceTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.weights = self.root / 'models' / 'checkpoints' / 'melanoma_resnet50_nb.pth'
        self.label_map = self.root / 'models' / 'label_maps' / 'label_map_nb.json'
        self.temp_json = self.root / 'models' / 'checkpoints' / 'temperature.json'
        self.op_json = self.root / 'models' / 'checkpoints' / 'operating_points.json'
        # Pick a known sample if present
        self.sample = self.root / 'data' / 'ds' / 'img' / 'ISIC_0027990.jpg'

    def test_can_load_and_predict(self):
        if not self.weights.exists() or not self.label_map.exists() or not self.sample.exists():
            self.skipTest('Required artifacts or sample image not found; skipping smoke test.')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, label_map = load_model(str(self.weights), str(self.label_map), device)

        pil_img, inp = preprocess_image(str(self.sample), img_size=224)
        T_val = load_temperature(str(self.temp_json))
        logits, probs = predict(model, inp, device, temperature=T_val)
        self.assertEqual(probs.ndim, 2)
        self.assertEqual(probs.shape[0], 1)
        self.assertEqual(probs.shape[1], len(label_map))

        op = load_operating_points(str(self.op_json))
        if op is not None:
            mel_idx = int(op.get('class_index', label_map.get('melanoma', -1)))
            self.assertTrue(mel_idx >= 0)
            mel_prob = float(probs.squeeze(0)[mel_idx].cpu().numpy())
            self.assertTrue(0.0 <= mel_prob <= 1.0)


if __name__ == '__main__':
    unittest.main()
