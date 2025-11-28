# 1. Introduction

## 1.1 Background and Motivation

Melanoma represents one of the most aggressive forms of skin cancer, accounting for the majority of skin cancer-related deaths despite comprising only a small fraction of cases. The American Cancer Society estimates that melanoma mortality exceeds 8,000 deaths annually in the United States alone, with incidence rates continuing to rise globally. However, when detected early, melanoma exhibits five-year survival rates exceeding 99%, dropping dramatically to 30% after metastasis. This stark contrast underscores the critical importance of early detection in melanoma management.

Current clinical practice relies heavily on visual examination of skin lesions, often aided by dermoscopy—a non-invasive imaging technique that illuminates subsurface skin structures invisible to the naked eye. While dermoscopy significantly improves diagnostic accuracy compared to unaided visual inspection, effective interpretation requires specialized training that many primary care physicians lack. Dermatologists typically develop this expertise through years of practice, building pattern recognition skills that remain difficult to codify or transfer systematically.

The shortage of trained dermatologists, particularly in rural and underserved areas, creates a bottleneck in access to expert melanoma screening. Patients in these regions often face long wait times for specialist referrals, during which lesions may progress. Even in well-resourced settings, the subjective nature of visual diagnosis leads to inter-observer variability, with studies reporting diagnostic agreement rates among dermatologists ranging from 60% to 80% for borderline cases.

Recent advances in deep learning have demonstrated remarkable capabilities in image recognition tasks, often matching or exceeding human expert performance. Convolutional neural networks (CNNs) trained on large medical imaging datasets have shown particular promise in dermatology, where the diagnostic task fundamentally involves visual pattern recognition. These models learn hierarchical feature representations directly from data, potentially capturing subtle patterns that elude manual specification or even conscious expert recognition.

However, deploying machine learning in clinical settings introduces challenges beyond raw classification accuracy. Healthcare providers require not just predictions, but confidence estimates that accurately reflect uncertainty. They need explanations that connect model decisions to established medical knowledge, enabling them to verify reasoning and integrate AI recommendations with other clinical information. The model must operate reliably across the range of image qualities and patient demographics encountered in practice, rather than only on carefully curated research datasets.

## 1.2 Problem Statement

This thesis addresses the challenge of developing a practically deployable melanoma detection system that balances multiple requirements: high classification accuracy, calibrated confidence estimates suitable for clinical decision-making, interpretable explanations that align with dermatological criteria, and accessible deployment through a user-friendly interface.

Specifically, I investigate three core questions:

1. **Model Selection**: Among contemporary deep learning architectures, which provides the best balance of accuracy, calibration quality, and inference speed for melanoma classification from dermoscopic images?

2. **Clinical Reliability**: How can we transform raw model outputs into calibrated probabilities with clinically-appropriate operating thresholds that reflect real-world diagnostic requirements?

3. **Explainability Integration**: Can attention-based visualization techniques provide diagnostically meaningful explanations that correlate with established clinical criteria, rather than merely highlighting discriminative but clinically opaque patterns?

## 1.3 Approach Overview

I developed an integrated system that addresses these questions through systematic experimentation and engineering. The approach comprises several key components:

**Dataset and Preprocessing**: I utilized the HAM10000 dataset, containing 10,013 dermoscopic images across seven diagnostic categories including melanoma, melanocytic nevi, basal cell carcinoma, and other lesion types. The dataset reflects real-world diagnostic challenges, including class imbalance (67% benign nevi, 11% melanoma) and variability in image quality and acquisition conditions. Images underwent standardized preprocessing including resizing to 224×224 pixels and normalization using ImageNet statistics, with data augmentation applied during training to improve generalization.

**Architecture Comparison**: I trained and evaluated four modern architectures representing different design philosophies: ResNet-50 (residual connections for deep networks), EfficientNet-B3 (compound scaling with neural architecture search), DenseNet-121 (dense connectivity patterns), and Vision Transformer ViT-B/16 (attention mechanisms without convolution). Each model was trained from ImageNet-pretrained weights for 20 epochs using Adam optimization with identical hyperparameters, ensuring fair comparison.

**Calibration and Thresholding**: Raw model outputs underwent temperature scaling to improve probability calibration, validated using expected calibration error and reliability diagrams. For melanoma detection specifically, I established operating thresholds that achieve 95% specificity, balancing sensitivity against false positive rates appropriate for a screening context where unnecessary biopsies carry costs but missed melanomas carry risks.

**Explainability Implementation**: I integrated Gradient-weighted Class Activation Mapping (Grad-CAM) to generate attention heatmaps highlighting image regions most influential to model predictions. These visualizations overlay on original dermoscopic images, enabling clinicians to verify that models focus on diagnostically relevant features (asymmetry, border irregularity, color variation) rather than artifacts or irrelevant image regions.

**Deployment Interface**: The complete system packages into an interactive web application using Gradio, allowing users to upload dermoscopic images and receive immediate feedback including predicted probabilities, attention heatmaps, and AI-generated explanations that contextualize predictions with relevant medical knowledge.

## 1.4 Key Contributions

This work makes several contributions to the intersection of deep learning and medical imaging:

1. **Systematic Architecture Comparison**: While many studies report results for individual models, this work provides direct comparison of four architectures under identical training conditions, dataset splits, and evaluation protocols. This reveals performance differences attributable to architectural choices rather than experimental variations.

2. **Clinical Calibration Framework**: The integration of temperature scaling with specificity-constrained thresholds demonstrates how to bridge the gap between machine learning performance metrics and clinical decision requirements. This framework generalizes to other medical screening applications where operating points must balance multiple clinical considerations.

3. **Validated Explainability**: Rather than treating explainability as an afterthought, the system design incorporates Grad-CAM attention maps as a core feature, validated against known dermatological criteria. This demonstrates that modern deep learning models can learn medically-interpretable feature representations when trained on appropriate data.

4. **Integrated Deployment**: Many research systems remain proof-of-concept demonstrations. This work provides a complete, documented implementation including data pipelines, trained models, calibration procedures, and user interface, ready for potential clinical pilot deployment or adaptation to related domains.

5. **Reproducible Methodology**: All code, model configurations, and training procedures are documented in detail, enabling reproduction of results and facilitating future work that builds on or extends this approach.

## 1.5 Thesis Organization

The remainder of this thesis is organized as follows: Chapter 2 reviews relevant background on melanoma diagnosis, deep learning in medical imaging, and explainable AI techniques. Chapter 3 details the methodology including dataset characteristics, model architectures, training procedures, calibration methods, and explainability implementation. Chapter 4 presents experimental results including architecture comparison, calibration analysis, and qualitative evaluation of attention maps. Chapter 5 discusses findings in clinical context, compares results to related work, and examines limitations and potential extensions. Chapter 6 concludes with a summary of contributions and directions for future research.
