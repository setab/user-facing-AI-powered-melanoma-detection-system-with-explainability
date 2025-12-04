# Melanoma Detection System - Data Flow Diagram

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MELANOMA DETECTION SYSTEM                          │
│                     Data Flow and Component Interaction                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                              1. DATA INPUT LAYER                             │
└──────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐         ┌─────────────────┐
    │  User Uploads   │         │  CLI Interface  │
    │ Dermoscopic Img │         │   (Optional)    │
    └────────┬────────┘         └────────┬────────┘
             │                           │
             └───────────┬───────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Image Validation  │
              │  - Format check     │
              │  - Size check       │
              └──────────┬──────────┘
                         │
                         ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│                         2. PREPROCESSING PIPELINE                            │
└──────────────────────────────────────────────────────────────────────────────┘

              ┌──────────────────────┐
              │  Image Preprocessing │
              │  (src/config.py)     │
              └──────────┬───────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌────────┐     ┌──────────┐    ┌─────────┐
    │ Resize │     │Normalize │    │  Color  │
    │224×224 │────▶│ImageNet  │───▶│  Space  │
    │ pixels │     │  Stats   │    │   RGB   │
    └────────┘     └──────────┘    └────┬────┘
                                         │
                                         ▼
                               ┌──────────────────┐
                               │  Tensor Creation │
                               │  [1, 3, 224,224] │
                               └─────────┬────────┘
                                         │
                                         ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│                      3. MODEL INFERENCE LAYER                                │
└──────────────────────────────────────────────────────────────────────────────┘

                        ┌──────────────────┐
                        │  Model Selection │
                        │  (EfficientNet)  │
                        └────────┬─────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
          ┌──────────────┐ ┌──────────┐ ┌──────────┐
          │   ResNet-50  │ │EfficientNet│DenseNet │
          │  25M params  │ │  12M params│ 8M params│
          │ 87.27% acc   │ │ 89.22% acc │86.12% acc│
          └──────┬───────┘ └─────┬──────┘└────┬─────┘
                 │               │             │
                 └───────────────┼─────────────┘
                                 │
                                 ▼
                    ┌───────────────────────┐
                    │   Forward Pass        │
                    │   - Conv layers       │
                    │   - Feature extraction│
                    │   - Global pooling    │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │   Raw Logits Output   │
                    │   [7 class scores]    │
                    │   z₁, z₂, ..., z₇     │
                    └──────────┬────────────┘
                               │
                               ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│                    4. CALIBRATION & THRESHOLDING                             │
└──────────────────────────────────────────────────────────────────────────────┘

                    ┌───────────────────────┐
                    │ Temperature Scaling   │
                    │   T = 1.6465          │
                    │   P = exp(z/T)        │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │  Calibrated Softmax   │
                    │  7-class probabilities│
                    │  [nv,mel,bkl,bcc,...] │
                    └──────────┬────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
        ┌──────────────┐ ┌──────────┐ ┌──────────────┐
        │   All 7      │ │ Melanoma │ │  Operating   │
        │   Classes    │ │Probability│ │  Threshold   │
        │  Displayed   │ │ Extracted │ │  @ 95% spec  │
        └──────┬───────┘ └─────┬────┘ └──────┬───────┘
               │               │              │
               │               ▼              │
               │      ┌─────────────────┐    │
               │      │  Threshold      │    │
               │      │  Comparison     │◀───┘
               │      │  P(mel) > 0.219 │
               │      └────────┬────────┘
               │               │
               └───────────────┼───────────────┐
                               │               │
                               ▼               ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│                      5. EXPLAINABILITY LAYER                                 │
└──────────────────────────────────────────────────────────────────────────────┘

                    ┌───────────────────────┐
                    │   Grad-CAM Module     │
                    │  (src/inference/xai)  │
                    └──────────┬────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
        ┌──────────────┐ ┌──────────┐ ┌──────────────┐
        │   Forward    │ │ Backward │ │   Gradient   │
        │  Pass Store  │ │   Pass   │ │ Computation  │
        │ Activations  │ │  ∂y/∂A   │ │ α = 1/Z·Σ∇   │
        └──────┬───────┘ └─────┬────┘ └──────┬───────┘
               │               │              │
               └───────────────┼──────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │  Weighted Combination │
                    │  L = ReLU(Σ αₖ·Aᵏ)    │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │  Heatmap Generation   │
                    │  - Upsample to 224×224│
                    │  - Apply colormap     │
                    │  - Overlay on image   │
                    └──────────┬────────────┘
                               │
                               ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│                      6. AI EXPLANATION GENERATION                            │
└──────────────────────────────────────────────────────────────────────────────┘

                    ┌───────────────────────┐
                    │ Explanation Generator │
                    │(src/serve_gradio.py)  │
                    └──────────┬────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
        ┌──────────────┐ ┌──────────┐ ┌──────────────┐
        │  Diagnosis   │ │   ABCDE  │ │ Clinical     │
        │   Analysis   │ │ Criteria │ │Recommendation│
        │ w/ Confidence│ │ Matching │ │   Based on   │
        │              │ │          │ │  Risk Level  │
        └──────┬───────┘ └─────┬────┘ └──────┬───────┘
               │               │              │
               └───────────────┼──────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │  Natural Language     │
                    │  Explanation Text     │
                    │  (Generated)          │
                    └──────────┬────────────┘
                               │
                               ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│                       7. OUTPUT & DISPLAY LAYER                              │
└──────────────────────────────────────────────────────────────────────────────┘

                    ┌───────────────────────┐
                    │  Gradio Web Interface │
                    │   (serve_gradio.py)   │
                    └──────────┬────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │ Probability  │    │  Grad-CAM    │    │     AI       │
  │   Display    │    │   Heatmap    │    │ Explanation  │
  │              │    │              │    │              │
  │ mel: 15.3%   │    │ [Overlay]    │    │ "This lesion │
  │ nv:  71.2%   │    │              │    │  shows..."   │
  │ bkl: 8.1%    │    │              │    │              │
  │ ...          │    │              │    │              │
  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
         │                   │                    │
         └───────────────────┼────────────────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │  Interactive Chat  │
                  │    Q&A System      │
                  └──────────┬─────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │   User Feedback    │
                  │  (Medical Review)  │
                  └────────────────────┘
```

---

## Detailed Component Data Flows

### Flow 1: Training Pipeline (Offline)

```
HAM10000 Dataset (10,013 images)
        │
        ├─→ Build Metadata (data/build_metadata.py)
        │           │
        │           ├─→ HAM10000_metadata.csv
        │           └─→ Label mappings
        │
        ├─→ Train/Val Split (85/15)
        │           │
        │           ├─→ train_split.csv (8,511 images)
        │           └─→ val_split.csv (1,502 images)
        │
        └─→ Training Loop (src/training/compare_models.py)
                    │
                    ├─→ Data Augmentation
                    │   - Random flips
                    │   - Rotation ±30°
                    │   - Color jitter
                    │
                    ├─→ Forward Pass (4 models × 20 epochs)
                    │
                    ├─→ Loss Computation (CrossEntropy)
                    │
                    ├─→ Backward Pass (Adam optimizer)
                    │
                    ├─→ Model Checkpoints
                    │   - resnet50_checkpoint.pth (91 MB)
                    │   - efficientnet_b3_checkpoint.pth (42 MB)
                    │   - densenet121_checkpoint.pth (28 MB)
                    │   - vit_b_16_checkpoint.pth (328 MB)
                    │
                    ├─→ Temperature Calibration
                    │   - Optimize T on validation set
                    │   - Save temperature.json
                    │
                    ├─→ Operating Point Selection
                    │   - Find threshold @ 95% specificity
                    │   - Save operating_points.json
                    │
                    └─→ Results & Metrics
                        - comparison_results.json
                        - Visualizations (8 figures)
```

### Flow 2: Inference Pipeline (Real-time)

```
User Image Upload
        │
        ▼
[Preprocessing]
  - Resize: Original → 224×224 pixels
  - Normalize: RGB values with ImageNet stats
  - Convert: PIL Image → PyTorch Tensor [1,3,224,224]
        │
        ▼
[Model Loading]
  - Load checkpoint: efficientnet_b3_checkpoint.pth
  - Load temperature: T = 1.6465
  - Load threshold: 0.2191 (melanoma @ 95% spec)
  - Set model.eval() mode
        │
        ▼
[Forward Inference]
  - Input: [1, 3, 224, 224] tensor
  - Conv layers: Feature extraction
  - Pooling: Global average pooling
  - FC layer: [1, 7] logits
  - Time: ~10.06 ms
        │
        ▼
[Calibration]
  - Raw logits: [z₁, z₂, ..., z₇]
  - Temperature scale: zᵢ/T
  - Softmax: P(class) = exp(zᵢ/T) / Σ exp(zⱼ/T)
  - Output: 7 calibrated probabilities
        │
        ├─→ [All Classes Display]
        │   - nv: 0.712 (71.2%)
        │   - mel: 0.153 (15.3%)
        │   - bkl: 0.081 (8.1%)
        │   - bcc: 0.032 (3.2%)
        │   - akiec: 0.015 (1.5%)
        │   - vasc: 0.005 (0.5%)
        │   - df: 0.002 (0.2%)
        │
        └─→ [Melanoma Decision]
            - Extract P(melanoma)
            - Compare: P(mel) > 0.2191?
            - Decision: High/Low risk
        │
        ▼
[Grad-CAM Generation]
  - Hook final conv layer
  - Forward pass (store activations)
  - Backward from predicted class
  - Compute gradients: ∂y^c/∂A
  - Weight activations: α_k = (1/Z)·Σ∇
  - Combine: L = ReLU(Σ α_k·A^k)
  - Upsample: 7×7 → 224×224
  - Colormap: Jet (blue→red)
  - Overlay: 40% transparency
        │
        ▼
[Explanation Generation]
  - Analyze probabilities
  - Map to ABCDE criteria
  - Generate clinical context
  - Formulate recommendation
  - Create natural language text
        │
        ▼
[User Display]
  - Probability bar chart
  - Grad-CAM overlay image
  - AI explanation text
  - Interactive chat available
```

### Flow 3: Web Interface Interaction

```
User Opens Gradio Interface (http://localhost:7860)
        │
        ▼
[Homepage Loads]
  - Image upload component
  - Example images available
  - Medical disclaimer shown
        │
        ▼
[User Uploads Image]
  - Image → base64 encoding
  - Send to backend
        │
        ▼
[Backend Processing]
  - Receive image
  - predict_and_explain()
  - Return: {probs, heatmap, explanation}
        │
        ▼
[Display Results]
  - Update probability chart
  - Show Grad-CAM overlay
  - Display AI explanation
        │
        ▼
[Optional: User Asks Question]
  - Chat interface activated
  - process_qa_answer()
  - Context: {image, prediction, probs}
  - Generate contextual answer
        │
        ▼
[User Reviews & Downloads]
  - Can download heatmap
  - Can copy explanation
  - Can try another image
```

---

## Data Structures at Each Stage

### Stage 1: Raw Input
```python
{
    "format": "JPEG/PNG",
    "size": "variable (600×450 typical)",
    "channels": 3,
    "color_space": "RGB",
    "dtype": "uint8",
    "range": [0, 255]
}
```

### Stage 2: Preprocessed Tensor
```python
{
    "shape": [1, 3, 224, 224],
    "dtype": "torch.float32",
    "device": "cuda",
    "normalized": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
}
```

### Stage 3: Model Output (Raw)
```python
{
    "logits": torch.Tensor([2.1, 1.5, -0.3, -1.2, -2.1, -3.5, -4.1]),
    "shape": [1, 7],
    "classes": ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
}
```

### Stage 4: Calibrated Probabilities
```python
{
    "nv": 0.712,      # melanocytic nevi
    "mel": 0.153,     # melanoma
    "bkl": 0.081,     # benign keratosis
    "bcc": 0.032,     # basal cell carcinoma
    "akiec": 0.015,   # actinic keratoses
    "vasc": 0.005,    # vascular lesions
    "df": 0.002,      # dermatofibroma
    "sum": 1.000,     # properly normalized
    "temperature": 1.6465,
    "calibrated": True
}
```

### Stage 5: Grad-CAM Output
```python
{
    "heatmap": np.ndarray,
    "shape": [224, 224],
    "dtype": "float32",
    "range": [0.0, 1.0],
    "overlay": PIL.Image,
    "overlay_alpha": 0.4,
    "colormap": "jet"
}
```

### Stage 6: Final Output Package
```python
{
    "prediction": "melanoma",
    "confidence": 0.153,
    "all_probabilities": {...},
    "grad_cam_overlay": PIL.Image,
    "ai_explanation": "This lesion shows...",
    "recommendation": "Further examination recommended",
    "risk_level": "moderate",
    "inference_time_ms": 10.06,
    "model_used": "efficientnet_b3"
}
```

---

## Performance Metrics at Each Stage

| Stage | Metric | Value |
|-------|--------|-------|
| **Preprocessing** | Time per image | ~2-3 ms |
| **Model Inference** | Forward pass | ~10 ms |
| **Calibration** | Computation | <1 ms |
| **Grad-CAM** | Generation | ~15 ms |
| **Explanation** | Text generation | ~5 ms |
| **Total Pipeline** | End-to-end | ~32 ms |
| **Web Response** | User-perceived | ~50-100 ms |

---

## Error Handling & Edge Cases

```
Input Validation
    ├─→ Invalid format → Error: "Please upload JPEG/PNG"
    ├─→ Too large → Resize automatically
    ├─→ Too small → Error: "Image too small (<100px)"
    ├─→ Corrupted → Error: "Cannot read image"
    └─→ Wrong channels → Convert to RGB

Model Errors
    ├─→ OOM (out of memory) → Reduce batch size
    ├─→ CUDA unavailable → Fallback to CPU
    ├─→ Model not found → Error: "Please train model first"
    └─→ Incompatible weights → Error: "Model version mismatch"

Output Validation
    ├─→ Probabilities don't sum to 1 → Renormalize
    ├─→ NaN in output → Error: "Numerical instability"
    ├─→ Grad-CAM fails → Return prediction without heatmap
    └─→ Explanation timeout → Return generic explanation
```

---

## System Boundaries & Interfaces

### External Inputs
- User-uploaded dermoscopic images (JPEG/PNG)
- Optional: CLI commands
- Optional: API requests (if deployed)

### External Outputs
- Web interface display (HTML/CSS/JS via Gradio)
- Downloadable images (Grad-CAM overlays)
- Text outputs (predictions, explanations)
- Optional: JSON API responses

### Internal Data Storage
- **Persistent**: Model checkpoints, calibration params, dataset metadata
- **Temporary**: Uploaded images (in-memory), computed tensors (GPU memory)
- **Cached**: Loaded models (RAM), preprocessing transforms

### Dependencies
- **Input**: HAM10000 dataset, ImageNet pretrained weights
- **Processing**: PyTorch, torchvision, NumPy, PIL
- **Output**: Gradio, Matplotlib (for visualizations)
- **Optional**: CUDA (GPU acceleration)

---

*This data flow diagram maps the complete journey of data through your melanoma detection system, from raw image upload to final clinical output with explanations.*
