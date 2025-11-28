# 2. Background and Related Work

## 2.1 Clinical Context: Melanoma Diagnosis

### 2.1.1 Medical Significance

Melanoma arises from melanocytes, the pigment-producing cells in skin, and exhibits aggressive growth patterns that enable rapid metastasis. Unlike basal cell carcinoma and squamous cell carcinoma—the more common but less lethal skin cancers—melanoma frequently spreads to lymph nodes and distant organs, making late-stage treatment challenging. The five-year survival rate demonstrates dramatic stage-dependence: localized melanoma shows 99% survival, regional lymph node involvement drops this to 68%, and distant metastasis reduces survival to 30%.

This stage-dependent prognosis makes early detection the single most important factor in melanoma outcomes. The American Cancer Society's guidelines recommend regular skin self-examinations and professional screening, particularly for high-risk individuals with fair skin, family history, or numerous atypical moles. However, early-stage melanoma often appears similar to benign lesions, requiring trained expertise for differentiation.

### 2.1.2 The ABCDE Criteria

Dermatologists employ the ABCDE criteria as a systematic framework for melanoma assessment:

- **Asymmetry**: Benign lesions typically exhibit bilateral symmetry, while melanomas often grow irregularly, producing asymmetric shapes when divided by any axis.
- **Border**: Benign nevi usually have smooth, well-defined borders, whereas melanomas frequently display notched, scalloped, or poorly defined edges.
- **Color**: Uniform pigmentation characterizes most benign lesions, while melanomas often exhibit multiple colors or shades including black, brown, red, white, or blue.
- **Diameter**: Melanomas commonly exceed 6mm in diameter, though this criterion alone lacks specificity as many benign lesions also exceed this threshold.
- **Evolution**: Changes in size, shape, color, or symptomatology (itching, bleeding) warrant concern, as melanomas typically evolve while stable lesions remain static.

These criteria provide heuristic guidance rather than deterministic rules. Many melanomas violate one or more criteria, particularly in early stages, while many benign lesions exhibit ABCDE features. Expert dermatologists integrate these observations with additional factors including patient history, lesion location, and subtle pattern recognition developed through experience.

### 2.1.3 Dermoscopy

Dermoscopy (also called dermatoscopy or epiluminescence microscopy) employs magnification and specialized lighting to visualize subsurface skin structures invisible to unaided examination. The technique reduces surface reflection and enhances contrast of pigmented structures, revealing patterns such as:

- **Pigment networks**: Regular networks suggest benign nevi, while atypical networks with irregular holes and thick lines indicate melanoma risk.
- **Dots and globules**: Regular distribution suggests benign lesions; irregular or peripheral accumulation raises melanoma concern.
- **Vascular patterns**: Benign lesions show orderly vessel arrangements, while melanomas often display irregular, polymorphic vessels.
- **Regression structures**: White scar-like areas or blue-gray zones may indicate spontaneous regression, associated with melanoma.

Meta-analyses demonstrate that dermoscopy improves diagnostic accuracy by 10-15% compared to naked-eye examination. However, effective dermoscopy interpretation requires training and experience. Studies show that experienced dermatologists achieve sensitivities of 75-95% for melanoma detection using dermoscopy, while less experienced practitioners show considerably lower performance, highlighting the expertise-dependent nature of visual diagnosis.

## 2.2 Deep Learning for Medical Imaging

### 2.2.1 Convolutional Neural Networks

Convolutional neural networks revolutionized computer vision by learning hierarchical feature representations directly from data rather than relying on hand-engineered features. The key architectural components include:

**Convolutional Layers**: Apply learned filters that detect local patterns in images. Early layers typically learn edge and texture detectors, while deeper layers combine these into higher-level structures. The convolution operation preserves spatial relationships, making CNNs particularly effective for visual tasks.

**Pooling Layers**: Reduce spatial dimensions while retaining important features, providing translation invariance and reducing computation. Max pooling selects the maximum activation in local regions, effectively highlighting the strongest feature responses.

**Activation Functions**: Introduce nonlinearity essential for learning complex representations. ReLU (Rectified Linear Unit) and its variants have become standard, enabling efficient gradient-based optimization while avoiding vanishing gradient problems that plagued earlier architectures.

**Fully Connected Layers**: Combine features from convolutional layers to produce final predictions. In classification tasks, the final layer typically uses softmax activation to produce probability distributions over classes.

### 2.2.2 Transfer Learning

Training deep networks from scratch requires massive datasets and substantial computational resources. Transfer learning addresses this by initializing models with weights pretrained on large general-purpose datasets like ImageNet (14 million images, 1000 classes), then fine-tuning on specific target tasks.

This approach works because early convolutional layers learn general features (edges, textures, colors) applicable across domains, while later layers adapt to task-specific patterns. For medical imaging, transfer learning has proven remarkably effective despite distribution differences between natural images and medical scans, consistently outperforming training from random initialization particularly when labeled medical data is limited.

### 2.2.3 Architecture Evolution

**ResNet (Residual Networks)**: Introduced skip connections that add layer inputs to their outputs, enabling training of very deep networks by addressing gradient degradation. ResNet-50 achieved breakthrough performance on ImageNet and became a standard baseline for computer vision tasks.

**DenseNet (Densely Connected Networks)**: Extended residual connections by concatenating feature maps from all previous layers to each layer, encouraging feature reuse and improving gradient flow. DenseNet achieves strong performance with fewer parameters than comparable ResNets.

**EfficientNet**: Applied neural architecture search to optimize network depth, width, and resolution simultaneously using a compound scaling method. EfficientNet models achieve state-of-the-art accuracy with significantly fewer parameters and computations than previous architectures.

**Vision Transformers**: Adapted the transformer architecture from natural language processing to vision by treating images as sequences of patches. ViT models achieve competitive or superior performance to CNNs when trained on sufficient data, though they typically require more training examples or pretraining to reach peak performance.

## 2.3 Deep Learning in Dermatology

### 2.3.1 Landmark Studies

Esteva et al. (2017) published a seminal study in *Nature* demonstrating that a deep CNN achieved dermatologist-level accuracy for skin cancer classification. Their model, trained on 129,450 clinical images, matched performance of 21 board-certified dermatologists on melanoma detection and keratinocyte carcinoma classification. This work established proof-of-concept that deep learning could reach expert-level performance on dermatological diagnosis from images alone.

Haenssle et al. (2018) conducted a prospective comparison in *Annals of Oncology*, showing that a CNN trained on 12,378 dermoscopic images outperformed 58 dermatologists (including 30 experts with >5 years dermoscopy experience) on a test set of 100 images. The CNN achieved 95% sensitivity and 82.5% specificity, compared to 86.6% and 71.3% for the dermatologists on average. Performance varied substantially among individual dermatologists, with sensitivities ranging from 72% to 93% and specificities from 60% to 91%, highlighting the difficulty and inter-observer variability of the task.

### 2.3.2 Public Datasets

Several large-scale dermoscopic image datasets have enabled deep learning research in dermatology:

**HAM10000 (Human Against Machine with 10000 training images)**: Released in 2018, this dataset contains 10,015 dermoscopic images representing seven common pigmented lesion categories, collected from two dermatology clinics over 20 years. All images include confirmed diagnoses from histopathology or expert consensus. The dataset exhibits realistic class imbalance and image quality variation, making it well-suited for developing systems intended for real-world deployment.

**ISIC Archive**: The International Skin Imaging Collaboration maintains a large public repository of dermoscopic images with standardized metadata and multiple challenge tasks. ISIC datasets have served as benchmarks for numerous competitions and publications, though annotation quality and diagnostic gold standards vary across subsets.

**BCN20000**: A dataset of 20,000 dermoscopic images from Hospital Clínic de Barcelona, representing eight diagnostic categories with histopathological confirmation. The larger size and institutional consistency make this dataset valuable for training, though it remains less commonly used as a standard benchmark than HAM10000 or ISIC.

### 2.3.3 Current Limitations

Despite promising results, several challenges limit clinical deployment of deep learning for dermatology:

**Dataset Bias**: Most training datasets come from populations with lighter skin tones and limited demographic diversity. Models may perform poorly on underrepresented groups, raising concerns about equitable access to AI-assisted diagnosis.

**Generalization**: Models trained on images from specific institutions or equipment often show performance degradation when applied to images from different sources, suggesting overfitting to dataset-specific characteristics rather than learning general diagnostic features.

**Clinical Integration**: High classification accuracy alone does not ensure clinical utility. Healthcare providers need calibrated probabilities, explanations, and interfaces that fit existing workflows. The path from research prototype to clinical tool remains challenging.

## 2.4 Model Calibration

### 2.4.1 The Calibration Problem

Neural networks trained with standard cross-entropy loss often produce overconfident predictions, particularly for incorrect classifications. A well-calibrated model produces probability estimates that match empirical frequencies: when the model assigns 80% probability to a diagnosis, approximately 80% of such predictions should be correct.

Calibration matters critically in medical applications where probability estimates inform clinical decisions. A miscalibrated model might assign 95% confidence to incorrect diagnoses or 60% confidence to correct ones, undermining provider trust and potentially leading to poor treatment decisions.

### 2.4.2 Temperature Scaling

Temperature scaling, introduced by Guo et al. (2017), provides a simple yet effective calibration method. The technique adds a single scalar parameter T (temperature) that divides logits before softmax:

$$P(y=k|x) = \frac{\exp(z_k/T)}{\sum_j \exp(z_j/T)}$$

where $z_k$ represents the model's logit for class k. Temperature T > 1 softens the probability distribution, reducing overconfidence, while T < 1 sharpens it. The optimal temperature is found by minimizing negative log-likelihood on a validation set.

Temperature scaling maintains the relative ordering of probabilities (the predicted class remains unchanged) while improving calibration. It requires minimal computation and only a single parameter, making it practical for deployment. Studies show temperature scaling consistently improves calibration across architectures and domains, often reducing expected calibration error by 50% or more.

### 2.4.3 Evaluation Metrics

**Expected Calibration Error (ECE)**: Partitions predictions into bins by confidence level, then computes the weighted average difference between confidence and accuracy within each bin. Lower ECE indicates better calibration.

**Maximum Calibration Error (MCE)**: Reports the maximum gap between confidence and accuracy across bins, highlighting worst-case calibration failures.

**Reliability Diagrams**: Plot predicted confidence against empirical accuracy for binned predictions. Well-calibrated models show points near the diagonal, while systematic deviations above or below indicate over- or under-confidence.

**Brier Score**: Measures mean squared error between predicted probabilities and true labels, combining calibration and refinement (discrimination) into a single metric. Lower scores indicate better probability estimates.

## 2.5 Explainable AI in Medical Imaging

### 2.5.1 The Black Box Problem

Deep neural networks learn complex, hierarchical representations distributed across millions of parameters. While this enables powerful pattern recognition, it creates opacity: even network designers cannot easily determine why a particular input produces a specific output. This "black box" problem raises concerns in high-stakes domains like medicine, where:

1. **Verification**: Clinicians need to verify that models focus on relevant image regions and features rather than spurious correlations or artifacts.
2. **Trust**: Healthcare providers require understanding to trust and responsibly integrate AI recommendations with other clinical information.
3. **Error Analysis**: When models fail, explanations help identify whether failures stem from image quality issues, model limitations, or edge cases requiring human judgment.
4. **Learning**: Explanations that align with medical knowledge can provide educational value, helping trainees learn diagnostic features.

### 2.5.2 Gradient-Based Attribution Methods

**Gradient Visualization**: Computes gradients of the output with respect to input pixels, highlighting which pixels most influence the prediction. However, raw gradients often produce noisy visualizations that lack spatial structure.

**Integrated Gradients**: Addresses gradient saturation by integrating gradients along a path from a baseline image to the actual input, providing more complete attribution. However, the method requires choosing an appropriate baseline and computing multiple gradient evaluations.

**SmoothGrad**: Adds random noise to the input and averages gradients over multiple noisy versions, reducing visual noise and highlighting more coherent patterns. This simple modification substantially improves gradient visualization interpretability.

### 2.5.3 Class Activation Mapping

**CAM (Class Activation Mapping)**: Zhou et al. (2016) introduced CAM for CNNs with global average pooling before the final classification layer. CAM weights spatial features by their importance to the predicted class, producing heatmaps that highlight discriminative image regions. However, CAM requires specific architectural constraints, limiting its applicability.

**Grad-CAM (Gradient-weighted Class Activation Mapping)**: Selvaraju et al. (2017) generalized CAM to any CNN architecture by using gradients flowing into the final convolutional layer to weight spatial activation maps:

$$L_{Grad-CAM} = ReLU\left(\sum_k \alpha_k A^k\right)$$

where $A^k$ represents activation maps and $\alpha_k = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A^k_{ij}}$ represents importance weights derived from gradients of class score $y^c$.

Grad-CAM produces coarse localization maps highlighting important regions while remaining class-discriminative. The method applies to any differentiable model and requires only a single forward and backward pass, making it computationally practical. Extensive validation shows Grad-CAM localizes relevant features across diverse vision tasks and architectures.

### 2.5.4 Medical Imaging Applications

Several studies have applied explainable AI to medical image classification:

- **Chest X-rays**: Grad-CAM visualizations for pneumonia detection often highlight lung infiltrates consistent with radiological findings, building clinician confidence in model predictions.

- **Diabetic Retinopathy**: Attention maps for retinal fundus images localize microaneurysms, hemorrhages, and exudates matching ophthalmologic assessment.

- **Brain MRI**: Tumor segmentation models coupled with attention visualization help verify that classifications utilize tumor characteristics rather than artifacts or scanner-specific patterns.

However, validation remains challenging: even when attention maps appear plausible to human observers, they may not reflect model reasoning. Recent work emphasizes the need for quantitative evaluation of explanation faithfulness and diagnostic relevance rather than purely subjective assessment.

## 2.6 Web-Based Medical AI Systems

Deploying research prototypes as accessible tools requires addressing software engineering challenges beyond model training. Gradio has emerged as a popular framework for rapidly building and sharing machine learning interfaces, particularly in research contexts. The library provides high-level abstractions for common UI components (image upload, sliders, text boxes) and automatically handles web serving, making it possible to create interactive demos with minimal code.

For medical AI systems, web interfaces must balance usability with appropriate caution. Clear disclaimers, probability displays rather than binary predictions, and integration of explanatory content help ensure providers use systems as decision support tools rather than autonomous diagnostic oracles. The interface design significantly impacts how clinicians interpret and act on model outputs, making human-computer interaction considerations integral to system development rather than afterthoughts.
