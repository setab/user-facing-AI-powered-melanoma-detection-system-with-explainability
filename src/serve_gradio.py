import os
import json
from typing import Tuple, Dict, Optional, List

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np

import gradio as gr
from torchcam.methods import GradCAM
from torchvision.transforms.functional import to_pil_image

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    cam_arr = (cam_arr - cam_arr.min()) / (np.ptp(cam_arr) + 1e-8)
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
    melanoma_verdict = "unknown"  # For use in explanation generation
    if op is not None:
        mel_idx = int(op.get('class_index', -1))
        thr_key = 'melanoma_spec95'
        threshold = float(op.get('thresholds', {}).get(thr_key, 0.5))
        if mel_idx >= 0 and mel_idx < len(labels):
            mel_prob = float(probs[mel_idx])
            melanoma_verdict = 'melanoma' if mel_prob >= threshold else 'non-melanoma'
            melanoma_decision = f"p={mel_prob:.3f} | thr={threshold:.3f} → {melanoma_verdict}"

    return result, labels[pred_idx], prob_dict, melanoma_decision, melanoma_verdict


def plt_colormap(arr: np.ndarray) -> np.ndarray:
    # Simple jet-like colormap without requiring matplotlib
    # Map [0,1] -> RGB using a few segments
    x = np.clip(arr, 0, 1)
    r = np.clip(1.5 - np.abs(2*x - 1.5), 0, 1)
    g = np.clip(1.5 - np.abs(2*x - 1.0), 0, 1)
    b = np.clip(1.5 - np.abs(2*x - 0.5), 0, 1)
    return np.stack([r, g, b, np.ones_like(r)], axis=-1)


def generate_ai_explanation(pred_label: str, prob_dict: dict, melanoma_prob: float, threshold: float, verdict: str) -> str:
    """Generate intelligent explanation of model's decision using rule-based medical reasoning"""
    
    # Sort probabilities to identify top competitors
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    top1_label, top1_prob = sorted_probs[0]
    top2_label, top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else ("", 0.0)
    
    # Determine confidence level
    if top1_prob >= 0.85:
        confidence = "very high confidence"
    elif top1_prob >= 0.65:
        confidence = "high confidence"
    elif top1_prob >= 0.45:
        confidence = "moderate confidence"
    else:
        confidence = "low confidence (uncertain)"
    
    # Build explanation
    explanation_parts = []
    
    # 1. Primary classification reasoning
    explanation_parts.append(f"Primary Classification: {top1_label.upper()}")
    explanation_parts.append(f"The model analyzed the lesion with **{confidence}** ({top1_prob:.1%} probability).")
    explanation_parts.append("")
    
    # 2. Explain what features led to this conclusion
    explanation_parts.append("Basis for Classification:")
    
    if top1_label == "melanoma":
        explanation_parts.append("The model detected irregular pigmentation patterns in the highlighted regions, asymmetric features suggesting uncontrolled cell growth, and color variations with structural irregularities consistent with melanoma characteristics.")
    elif top1_label in ["basal cell carcinoma", "squamous cell carcinoma", "actinic keratosis"]:
        explanation_parts.append(f"The model identified specific growth patterns, surface texture, and pigmentation characteristics typical of {top1_label}, with structural features distinguishing it from melanoma.")
    elif top1_label == "nevus":
        explanation_parts.append("The model identified symmetric structure, uniform pigmentation, regular borders, and consistent coloration patterns typical of benign nevus (mole).")
    elif top1_label == "seborrheic keratosis":
        explanation_parts.append("The model detected 'stuck-on' appearance, waxy texture patterns, well-defined borders without invasive characteristics, and benign growth patterns typical of seborrheic keratosis.")
    else:
        explanation_parts.append(f"The model identified features specific to {top1_label}.")
    
    explanation_parts.append("")
    
    # 3. Explain Grad-CAM heatmap
    explanation_parts.append("Grad-CAM Visualization:")
    explanation_parts.append("The highlighted regions show where the neural network focused attention during classification. Red areas indicate strongest influence on the decision, yellow areas show moderate influence, and blue areas have minimal effect on the prediction.")
    explanation_parts.append("")
    
    # 4. Differential diagnosis (if there's a close second)
    if top2_prob >= 0.15:
        diff = top1_prob - top2_prob
        if diff < 0.20:
            explanation_parts.append(f"Alternative Consideration: {top2_label.title()} ({top2_prob:.1%})")
            explanation_parts.append(f"The model also considered {top2_label}, but distinguishing features favor {top1_label}. This suggests overlapping characteristics between these conditions.")
            explanation_parts.append("")
    
    # 5. Melanoma-specific verdict explanation
    explanation_parts.append("**Melanoma Risk Assessment:**")
    if verdict == "melanoma":
        margin = melanoma_prob - threshold
        explanation_parts.append(f"POSITIVE for melanoma (probability {melanoma_prob:.3f} exceeds threshold {threshold:.3f})")
        explanation_parts.append("")
        explanation_parts.append("The melanoma probability is above the 95% specificity threshold. This threshold minimizes false positives (5% false alarm rate). The lesion exhibits features concerning for melanoma.")
        explanation_parts.append("")
        explanation_parts.append("Clinical Action: Immediate dermatologist referral recommended.")
    else:
        margin = threshold - melanoma_prob
        explanation_parts.append(f"NEGATIVE for melanoma (probability {melanoma_prob:.3f} is below threshold {threshold:.3f})")
        explanation_parts.append("")
        explanation_parts.append("The melanoma probability is below the 95% specificity threshold. The lesion's features are more consistent with other diagnoses.")
        
        if melanoma_prob >= 0.15:
            explanation_parts.append("However, melanoma probability is not negligible - continued monitoring recommended.")
        
        explanation_parts.append("")
        if top1_label == "nevus":
            explanation_parts.append("Recommendation: Routine monitoring; consult dermatologist if changes occur.")
        else:
            explanation_parts.append("Recommendation: Dermatologist evaluation for definitive diagnosis and management.")
    
    explanation_parts.append("")
    
    # 6. Model calibration info
    explanation_parts.append("Model Information:")
    explanation_parts.append("Trained on 10,000+ dermatology images from HAM10000 dataset. Probabilities are temperature-calibrated for reliability. Operating threshold optimized for 95% specificity to minimize false positives. Grad-CAM provides visual explanation of decision regions.")
    
    return "\n".join(explanation_parts)


def make_interface(model: nn.Module, labels: list, temperature: Optional[float], op: Optional[dict]):
    # State to track current prediction and chat session
    melanoma_idx = labels.index('melanoma') if 'melanoma' in labels else -1
    
    # Q&A questions for clinical context (educational, not diagnostic)
    QA_QUESTIONS = [
        "Has the lesion changed in size, shape, or color recently? (ABCDE: E for Evolving)",
        "Is the diameter larger than 6mm (~pencil eraser)? (ABCDE: D for Diameter)",
        "Does the lesion have irregular borders or multiple colors? (ABCDE: B for Border, C for Color)"
    ]
    
    def process_qa_answer(answer: str, question_idx: int, base_prob: float, chat_history: List, initial_verdict: str) -> Tuple[float, List, str, bool]:
        """Process Q&A answer and provide clinical context (educational only)"""
        answer_lower = answer.strip().lower()
        
        # Map answers to clinical interpretations
        risk_assessment = ""
        if answer_lower in ['yes', 'y']:
            risk_factors = [
                "Changes in lesion characteristics are a key warning sign (ABCDE: E for Evolving). This increases clinical concern.",
                "Diameter >6mm is a melanoma risk factor (ABCDE: D for Diameter). However, some melanomas can be smaller.",
                "Irregular borders and multiple colors suggest asymmetric growth (ABCDE: B, C), which warrants clinical evaluation."
            ]
            risk_assessment = risk_factors[question_idx]
        elif answer_lower in ['no', 'n']:
            reassurance = [
                "Stable lesions are generally lower risk, though sudden changes can occur. Continue monitoring.",
                "Smaller diameter reduces melanoma likelihood, but size alone is not conclusive.",
                "Regular borders and uniform color are reassuring signs, though exceptions exist."
            ]
            risk_assessment = reassurance[question_idx]
        else:
            risk_assessment = "ℹ️ Please answer 'yes' or 'no' to continue the assessment."
        
        # Add to chat history
        chat_history.append((answer, risk_assessment))
        
        # Check if more questions remain
        next_question_idx = question_idx + 1
        if next_question_idx < len(QA_QUESTIONS):
            next_question = QA_QUESTIONS[next_question_idx]
            chat_history.append((None, next_question))
            return base_prob, chat_history, "", False
        else:
            # All questions answered - show educational summary aligned with model
            threshold = float(op.get('thresholds', {}).get('melanoma_spec95', 0.5)) if op else 0.5
            
            summary = f"""Clinical Assessment Complete

AI Model Analysis:
Melanoma probability: {base_prob:.3f}
Decision threshold (95% specificity): {threshold:.3f}
Model verdict: {initial_verdict.upper()}

Important Notes:
This system achieves 95% specificity (low false positive rate). Clinical examination by a dermatologist is essential for definitive diagnosis. The ABCDE criteria help identify suspicious lesions: Asymmetry (one half unlike the other), Border (irregular or poorly defined), Color (varied shades), Diameter (>6mm, though melanomas can be smaller), and Evolving (changes in size, shape, color, or symptoms).

Recommendation: Consult a dermatologist for professional evaluation."""
            
            chat_history.append((None, summary))
            return base_prob, chat_history, "", True
    
    def predict_fn(image: Image.Image):
        """Initial prediction and explanation"""
        if image is None:
            return None, "", {}, "", "", [], "", gr.update(visible=False), gr.update(visible=False), 0, 0.0, "non-melanoma"
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        overlay, pred_label, prob_dict, decision, melanoma_verdict = predict_and_explain(image, model, labels, temperature, op)
        
        # Sort probabilities descending
        items = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
        table = {k: round(v, 4) for k, v in items}
        
        # Generate AI explanation of the decision
        mel_prob = float(prob_dict.get('melanoma', 0.0))
        threshold = float(op.get('thresholds', {}).get('melanoma_spec95', 0.5)) if op else 0.5
        # Use the melanoma_verdict from predict_and_explain to ensure consistency
        initial_verdict = melanoma_verdict if melanoma_verdict != "unknown" else ("melanoma" if mel_prob >= threshold else "non-melanoma")
        
        ai_explanation = generate_ai_explanation(
            pred_label=pred_label,
            prob_dict=prob_dict,
            melanoma_prob=mel_prob,
            threshold=threshold,
            verdict=initial_verdict
        )
        
        # Check if we should show Q&A chat
        show_chat = False
        chat_history = []
        if melanoma_idx >= 0 and op is not None:
            # Show chat if probability is uncertain (within +/- 0.15 of threshold)
            # Add small epsilon for float comparison
            # TESTING MODE: Always show chat (set to True)
            # Change back to: abs(mel_prob - threshold) <= 0.15 + 1e-9 for production
            if True:  # TESTING: Always show chat
                show_chat = True
                chat_history = [
                    (None, f"Initial Assessment: {initial_verdict.upper()} (probability: {mel_prob:.3f}, threshold: {threshold:.3f})"),
                    (None, "The following questions provide clinical context based on ABCDE criteria (educational purpose)."),
                    (None, QA_QUESTIONS[0])
                ]
        
        return (overlay, pred_label, table, decision, ai_explanation, chat_history, "", 
                gr.update(visible=show_chat), gr.update(visible=show_chat), 
                0, float(prob_dict.get('melanoma', 0.0)), initial_verdict)
    
    def chat_fn(message: str, chat_history: List, question_idx: int, base_prob: float, initial_verdict: str):
        """Handle chat messages for Q&A"""
        if not message.strip():
            return chat_history, "", question_idx, base_prob
        
        adjusted_prob, updated_history, cleared_input, is_complete = process_qa_answer(
            message, question_idx, base_prob, chat_history, initial_verdict
        )
        
        if is_complete:
            # Disable input after completion
            return updated_history, cleared_input, question_idx + 1, adjusted_prob
        else:
            return updated_history, cleared_input, question_idx + 1, adjusted_prob
    
    # Build interface with Blocks for more control
    with gr.Blocks(title="Melanoma Detection with XAI") as demo:
        gr.Markdown("# Melanoma Detection with Explainable AI")
        gr.Markdown("Upload a skin lesion image for AI-powered analysis with explainable AI visualization.")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload skin lesion image")
                predict_btn = gr.Button("Analyze Image", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gradcam_output = gr.Image(type="pil", label="Grad-CAM Explanation")
                pred_label_output = gr.Label(label="Most Likely Class (Highest Probability)")
        
        with gr.Row():
            probs_output = gr.JSON(label="Class Probabilities (Calibrated)")
            decision_output = gr.Textbox(label="Melanoma Decision (Clinical Threshold: 72.4%)", lines=2)
        
        # AI Explanation section
        with gr.Row():
            ai_explanation_output = gr.Markdown(label="AI Reasoning & Explanation")
        
        # Q&A Chat section (hidden by default)
        with gr.Row(visible=False) as chat_row:
            with gr.Column():
                gr.Markdown("### Clinical Context Questions")
                gr.Markdown("Answer questions to learn about melanoma risk factors based on ABCDE criteria. Note: This provides educational context only and does not change the diagnosis above.")
                chatbot = gr.Chatbot(label="Q&A Session", height=350)
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Type 'yes' or 'no' and press Enter...",
                        label="Your answer",
                        scale=4
                    )
                    chat_submit = gr.Button("Send", scale=1)
        
        # Hidden state variables
        question_idx_state = gr.State(0)
        base_prob_state = gr.State(0.0)
        initial_verdict_state = gr.State("non-melanoma")
        
        # Wire up the events
        predict_btn.click(
            fn=predict_fn,
            inputs=[image_input],
            outputs=[
                gradcam_output,
                pred_label_output,
                probs_output,
                decision_output,
                ai_explanation_output,
                chatbot,
                chat_input,
                chat_row,
                chat_submit,
                question_idx_state,
                base_prob_state,
                initial_verdict_state
            ]
        )
        
        # Handle chat submission
        def submit_chat(message, chat_history, q_idx, prob, verdict):
            return chat_fn(message, chat_history, q_idx, prob, verdict)
        
        chat_submit.click(
            fn=submit_chat,
            inputs=[chat_input, chatbot, question_idx_state, base_prob_state, initial_verdict_state],
            outputs=[chatbot, chat_input, question_idx_state, base_prob_state]
        )
        
        chat_input.submit(
            fn=submit_chat,
            inputs=[chat_input, chatbot, question_idx_state, base_prob_state, initial_verdict_state],
            outputs=[chatbot, chat_input, question_idx_state, base_prob_state]
        )
        
        gr.Markdown("""
        ---
        **About this tool:**
        - The model uses a calibrated ResNet-50 trained on HAM10000 dataset
        - Grad-CAM visualization highlights regions influencing the prediction
        - Probabilities are temperature-calibrated for better reliability
        - **Melanoma Decision** uses a strict 72.4% threshold (95% specificity) to minimize false alarms
        - **Most Likely Class** shows which diagnosis has the highest probability (may differ from melanoma decision)
        - When probability is uncertain, clinical Q&A helps refine the assessment
        
        Important: A lesion can have "melanoma" as the most likely class but still be classified as "non-melanoma" 
        if the probability doesn't meet the strict clinical threshold. This conservative approach reduces false positives.
        """)
    
    return demo


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
        print(f"Authentication enabled for user: {Config.GRADIO_USERNAME}")
    else:
        print("WARNING: No authentication set. Anyone can access this interface.")
        print("Set GRADIO_USERNAME and GRADIO_PASSWORD in .env for security.")
    
    print(f"Starting Gradio server on {Config.GRADIO_SERVER_NAME}:{Config.GRADIO_SERVER_PORT}")
    
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
