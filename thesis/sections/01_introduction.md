# 1. Introduction

## 1.1 Background and Motivation

Melanoma is one of the most aggressive skin cancers - despite being a relatively small fraction of skin cancer cases, it accounts for most of the deaths. The American Cancer Society estimates over 8,000 melanoma deaths annually just in the United States, and incidence rates keep rising globally. But here's the thing: when you catch melanoma early, five-year survival rates exceed 99%. After metastasis? That drops to 30%. This stark difference makes early detection absolutely critical.

Current clinical practice relies heavily on visual examination of skin lesions, often with dermoscopy - a non-invasive imaging technique that reveals subsurface skin structures you can't see with the naked eye. Dermoscopy definitely improves diagnostic accuracy compared to just looking at lesions, but it takes specialized training that most primary care physicians don't have. Dermatologists typically spend years building the pattern recognition skills needed to interpret these images effectively, and these skills are notoriously difficult to codify or teach systematically.

The shortage of trained dermatologists creates a real access problem, especially in rural and underserved areas. Patients in these regions often wait weeks or months for specialist referrals, during which lesions can progress. Even in well-resourced settings, the subjective nature of visual diagnosis leads to disagreement between experts - studies show diagnostic agreement among dermatologists ranging from 60-80% for borderline cases, which is concerning.

Recent work in deep learning has shown pretty remarkable results in image recognition, often matching or beating human experts. Convolutional neural networks trained on large medical imaging datasets have been particularly promising in dermatology, where the diagnostic task is fundamentally about visual pattern recognition. These models learn hierarchical features directly from data, potentially picking up on subtle patterns that are hard to specify manually or that experts might not even consciously recognize.

But deploying machine learning in clinical settings involves more than just classification accuracy. Healthcare providers need confidence estimates that actually reflect uncertainty, not just arbitrary numbers. They need explanations connecting model decisions to established medical knowledge so they can verify the reasoning and integrate AI recommendations with other clinical information. And the model needs to work reliably across the range of image qualities and patient demographics you see in practice, not just on carefully curated research datasets.

## 1.2 Problem Statement

My thesis tackles the challenge of building a melanoma detection system that's actually deployable in practice, balancing multiple requirements: high accuracy, calibrated confidence estimates for clinical decision-making, interpretable explanations aligned with dermatological criteria, and an accessible interface.

I investigate three main questions:

1. **Model Selection**: Among current deep learning architectures, which one gives the best balance of accuracy, calibration quality, and inference speed for melanoma classification from dermoscopic images?

2. **Clinical Reliability**: How do we transform raw model outputs into calibrated probabilities with clinically-appropriate thresholds that reflect real-world diagnostic requirements?

3. **Explainability**: Can attention-based visualization techniques provide diagnostically meaningful explanations that actually correlate with established clinical criteria, rather than just highlighting whatever discriminative patterns the model learned?

## 1.3 Approach Overview

I developed an integrated system addressing these questions through systematic experimentation. The approach has several key pieces:

**Dataset and Preprocessing**: I used the HAM10000 dataset with 10,013 dermoscopic images across seven diagnostic categories including melanoma, nevi, basal cell carcinoma, and other lesion types. The dataset reflects real-world challenges like class imbalance (67% benign nevi, only 11% melanoma) and variability in image quality and how they were captured. I preprocessed images to 224×224 pixels and normalized with ImageNet statistics, applying data augmentation during training to improve generalization.

**Architecture Comparison**: I trained four modern architectures representing different design philosophies: ResNet-50 (residual connections for very deep networks), EfficientNet-B3 (compound scaling from neural architecture search), DenseNet-121 (dense connectivity), and Vision Transformer ViT-B/16 (pure attention, no convolution). Each model started from ImageNet-pretrained weights and trained for 20 epochs with Adam using identical hyperparameters for fair comparison.

**Calibration and Thresholding**: Raw model outputs went through temperature scaling to improve probability calibration, which I validated using expected calibration error and reliability diagrams. For melanoma specifically, I set operating thresholds achieving 95% specificity, balancing sensitivity against false positive rates appropriate for screening where unnecessary biopsies have costs but missed melanomas have much worse consequences.

**Explainability**: I integrated Grad-CAM to generate attention heatmaps showing which image regions most influenced predictions. These visualizations overlay on the original images, letting clinicians verify that models focus on diagnostically relevant features (asymmetry, border irregularity, color variation) rather than artifacts or irrelevant stuff.

**Deployment**: The complete system packages into a web application using Gradio, allowing users to upload images and immediately get predicted probabilities, attention heatmaps, and AI-generated explanations contextualizing predictions with relevant medical knowledge.

## 1.4 Key Contributions

This work makes several contributions to the intersection of deep learning and medical imaging:

1. **Systematic Architecture Comparison**: While many studies report results for individual models, this work provides direct comparison of four architectures under identical training conditions, dataset splits, and evaluation protocols. This reveals performance differences attributable to architectural choices rather than experimental variations.

2. **Clinical Calibration Framework**: The integration of temperature scaling with specificity-constrained thresholds demonstrates how to bridge the gap between machine learning performance metrics and clinical decision requirements. This framework generalizes to other medical screening applications where operating points must balance multiple clinical considerations.

3. **Validated Explainability**: Rather than treating explainability as an afterthought, the system design incorporates Grad-CAM attention maps as a core feature, validated against known dermatological criteria. This demonstrates that modern deep learning models can learn medically-interpretable feature representations when trained on appropriate data.

4. **Integrated Deployment**: Many research systems remain proof-of-concept demonstrations. This work provides a complete, documented implementation including data pipelines, trained models, calibration procedures, and user interface, ready for potential clinical pilot deployment or adaptation to related domains.

5. **Reproducible Methodology**: All code, model configurations, and training procedures are documented in detail, enabling reproduction of results and facilitating future work that builds on or extends this approach.

## 1.5 Thesis Organization

The remainder of this thesis is organized as follows: Chapter 2 reviews relevant background on melanoma diagnosis, deep learning in medical imaging, and explainable AI techniques. Chapter 3 details the methodology including dataset characteristics, model architectures, training procedures, calibration methods, and explainability implementation. Chapter 4 presents experimental results including architecture comparison, calibration analysis, and qualitative evaluation of attention maps. Chapter 5 discusses findings in clinical context, compares results to related work, and examines limitations and potential extensions. Chapter 6 concludes with a summary of contributions and directions for future research.
