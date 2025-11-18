# Interactive Decision Support Layer for Melanoma Classification

You can wrap your **image classifier** with an *interactive decision support* layer that explains its focus (via Grad-CAM) and asks **2–4 targeted follow-up questions** when confidence is low. The system then updates the probability using a lightweight calibration model.

This is thesis-worthy, practical, and runs fine on your **16 GB GPU**.

---

## What You’ll Build

* **Core model**: ResNet50 (or stronger) trained on `ds/img` with labels from `ds/ann objects[].classTitle`.
* **Uncertainty gate**: Detect low confidence using entropy, max-probability, or margin.
* **Question policy**: A fixed set of clinically relevant yes/no or short categorical questions.
* **Probability update**: Logistic regression calibration on top of image logit + answers.
* **Explanation**: Grad-CAM heatmap + concise text combining image cues and user-provided factors.

---

## Step-by-Step

### 1. Train the Image Model

* Use class weights or Focal Loss.
* Enable AMP mixed precision.
* Apply early stopping by validation ROC-AUC.
* Save:

  * Best checkpoint
  * `label_map.json`

**Persist outputs per validation image:**

* `image_id`
* `y_true`
* `image_logit` (melanoma pre-softmax score)
* `p_melanoma`
* Optional: Grad-CAM map path

---

### 2. Define Uncertainty for Triggering Questions

* **Max probability**: low confidence if `0.35 ≤ max_p ≤ 0.75`.
* **Entropy**: `H = −Σ p log p`; uncertain if `H > 0.8` (for 7 classes).
* **Margin**: difference between top-1 and top-2 probabilities; low if `< 0.2`.

---

### 3. Choose 5–7 Questions

Pick small, clinically relevant items:

1. Age group: `<30`, `30–50`, `>50`
2. Lesion change: has it grown/changed recently? (yes/no)
3. Sun exposure: frequent/intense? (yes/no)
4. Family history of melanoma (yes/no)
5. Skin phototype: I–II / III–IV / V–VI
6. Lesion location: trunk / head & neck / extremities
7. Immune status or prior cancers (optional)

---

### 4. Encode Answers & Build Calibration Model

* Construct a validation table:

  * `y_true`
  * `image_logit`
  * Encoded answers/features

**Options if metadata missing:**

* Use available dataset fields (e.g., HAM10000 metadata).
* Simulate/pilot with a small annotated subset.

**Training:**

* Fit `LogisticRegression` (`scikit-learn`):

  ```
  y ~ image_logit + age + localization + sex
  ```
* Evaluate:

  * Expected Calibration Error (ECE)
  * ROC-AUC
  * Sensitivity at fixed specificity

---

### 5. Decide Question Order

* **Simple rule**: ask most predictive features first (by LR coefficients).
* **Advanced**: compute *expected information gain* → simulate both answers and choose the question that reduces entropy most.

---

### 6. Update Probability Online

1. Compute `p_img`, `logit_img`.
2. If low confidence → ask Q1, encode answer.
3. Recompute:

   ```
   p = sigmoid(b0 + b1*logit_img + Σ w_i*feat_i)
   ```
4. Repeat up to 3 questions or until confident (`p > 0.85` or `< 0.15`).

---

### 7. Explanations to Show

* **Visual**: Grad-CAM overlay with caption

  > “Model focused on darker, asymmetric regions; predicted melanoma probability 0.62.”

* **Text with answers**:

  * “Reported recent change: yes (increases risk).”
  * “Location: trunk (moderate effect).”
  * “After incorporating answers, probability adjusted to 0.74.”

---

## Validation for Thesis

* **Calibration**: compare ECE before vs after Q&A.

* **Decision curves**: net benefit vs threshold (with/without calibration).

* **Ablations**:

  * No questions vs 1 vs 3 questions
  * Fixed vs info-gain order
  * Image-only vs image+metadata

* **XAI Sanity**:

  * Randomization tests for Grad-CAM
  * Insertion/deletion curves
  * Overlap with lesion masks (if available)

---

## Guardrails & Disclaimers

* **Bias/Fairness**: check by demographic/localization.
* **Privacy**: don’t store user inputs without consent.
* **Disclaimer**: *“This tool is for research/educational use only; not a medical diagnosis.”*

---

## Code Hooks

### After Training

* Export per-image logits + metadata.
* Fit LR calibration model → save coefficients.

### Inference Flow

```python
p_img, logit_img = model(image)

if uncertain(p_img):
    ask(question)
    answer = encode(answer)
    p = sigmoid(b0 + b1*logit_img + Σ w_i*feat_i)
else:
    p = p_img
```

### Display

* Grad-CAM overlay
* Text summary with updated probability

---

## Sensible Defaults

* Trigger if `0.35 ≤ max_p ≤ 0.75` or entropy > 0.85.
* Max 3 questions (age group, location, recent change).
* Calibration: logistic regression (baseline: temperature scaling).
* Report: ROC-AUC, PR-AUC, sensitivity at 90% specificity, ECE.

---

👉 Now it reads clean and structured like a thesis chapter or design document.

Do you want me to also **turn this into a LaTeX template** (ready for thesis integration), or keep it in Markdown/Colab notebook style?
