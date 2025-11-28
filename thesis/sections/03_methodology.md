# 3. Methodology

## 3.1 Dataset

### 3.1.1 HAM10000 Overview

This work utilized the HAM10000 dataset (Tschandl et al., 2018), comprising 10,013 dermoscopic images of pigmented skin lesions collected from the dermatology practices at the Medical University of Vienna (Austria) and the Cliff Rosendahl Skin Cancer Practice (Australia) between 1999 and 2018. The dataset represents seven diagnostic categories:

1. **Melanocytic nevi (nv)**: 6,705 images (67.0%)
2. **Melanoma (mel)**: 1,112 images (11.1%)
3. **Benign keratosis-like lesions (bkl)**: 1,098 images (11.0%)
4. **Basal cell carcinoma (bcc)**: 514 images (5.1%)
5. **Actinic keratoses (akiec)**: 327 images (3.3%)
6. **Vascular lesions (vasc)**: 142 images (1.4%)
7. **Dermatofibroma (df)**: 115 images (1.1%)

All diagnoses were confirmed through one of four verification methods: histopathology (54.4% of cases), follow-up examination (16.8%), expert consensus (15.5%), or in-vivo confocal microscopy (13.3%). The dataset deliberately includes images of varying quality, orientation, and field-of-view to reflect real-world clinical acquisition conditions rather than idealized laboratory captures.

### 3.1.2 Class Imbalance

The dataset exhibits substantial class imbalance typical of medical screening scenarios, where benign lesions vastly outnumber malignant ones. Melanocytic nevi—benign moles—comprise 67% of images, while three categories (vascular lesions, dermatofibroma, actinic keratoses) each contribute less than 4%. This imbalance poses training challenges: without careful handling, models may achieve high overall accuracy by simply predicting common classes while failing on rare but clinically important categories like melanoma.

### 3.1.3 Train-Validation Split

I divided the dataset into training (85%) and validation (15%) sets using stratified random sampling to preserve class distributions. This produced:

- **Training set**: 8,511 images (5,699 nevi, 945 melanoma, 933 benign keratosis, 437 basal cell carcinoma, 278 actinic keratoses, 121 vascular, 98 dermatofibroma)
- **Validation set**: 1,502 images (1,006 nevi, 167 melanoma, 165 benign keratosis, 77 basal cell carcinoma, 49 actinic keratoses, 21 vascular, 17 dermatofibroma)

The same split was maintained across all experiments to ensure fair model comparison. I used a fixed random seed (42) to enable exact reproducibility of results. The validation set serves dual purposes: hyperparameter tuning (including temperature calibration) and final performance evaluation. While ideally a separate test set would be held out for unbiased evaluation, the limited dataset size necessitated this compromise, common in medical imaging research with moderate-sized datasets.

### 3.1.4 Data Preprocessing

All images underwent identical preprocessing regardless of model architecture:

1. **Resizing**: Images were resized to 224×224 pixels using bilinear interpolation. While original HAM10000 images vary in resolution (typically 600×450 pixels), 224×224 represents the standard input size for ImageNet-pretrained models and provides sufficient detail for dermoscopic pattern recognition.

2. **Normalization**: Pixel values were scaled to [0, 1] range then normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] for RGB channels respectively). This normalization matches the preprocessing used during ImageNet pretraining, facilitating effective transfer learning.

3. **Data Augmentation** (training only): To improve generalization and provide robustness to acquisition variations, training images underwent random augmentations applied on-the-fly:
   - Random horizontal and vertical flips (p=0.5 each)
   - Random rotation (up to ±30 degrees)
   - Random color jittering (brightness, contrast, saturation, hue varied by up to ±10%)
   - Random affine transformations (translation up to 10%, scale 0.9-1.1)

These augmentations simulate natural variations in image acquisition: lesion orientation, lighting conditions, and camera positioning. Validation images received only resizing and normalization (no augmentation) to provide consistent evaluation.

## 3.2 Model Architectures

I compared four architectures representing different design paradigms in modern computer vision:

### 3.2.1 ResNet-50

Residual Networks (He et al., 2016) introduced skip connections that add layer inputs to their outputs, enabling training of very deep networks by mitigating gradient degradation. ResNet-50 contains 50 layers organized into residual blocks:

$$y = F(x, \{W_i\}) + x$$

where $F$ represents stacked convolutional layers and the identity mapping $x$ provides a gradient highway for backpropagation. ResNet-50 uses bottleneck blocks with 1×1, 3×3, 1×1 convolutions that reduce then restore dimensionality, efficiently handling the 25 million parameters.

I initialized ResNet-50 with ImageNet pretrained weights and replaced the final 1000-way classification layer with a 7-way layer for the HAM10000 categories. All layers were fine-tuned during training rather than freezing early layers, as dermoscopic images differ substantially from natural images, potentially requiring adaptation even of low-level feature detectors.

### 3.2.2 EfficientNet-B3

EfficientNets (Tan & Le, 2019) apply neural architecture search to optimize network depth, width, and resolution simultaneously using a compound scaling method:

$$\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi$$

subject to $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ and $\alpha \geq 1, \beta \geq 1, \gamma \geq 1$, where $\phi$ controls overall resource availability.

EfficientNet-B3 scales the baseline architecture discovered through AutoML, achieving strong accuracy with 12 million parameters—less than half of ResNet-50. The architecture employs mobile inverted bottleneck convolutions with squeeze-and-excitation attention, enabling efficient feature learning. Like ResNet, I used ImageNet pretrained weights and adapted the final classification layer to 7 classes.

### 3.2.3 DenseNet-121

Densely Connected Networks (Huang et al., 2017) extend residual connections by concatenating feature maps from all previous layers to each layer:

$$x_\ell = H_\ell([x_0, x_1, ..., x_{\ell-1}])$$

where $[x_0, x_1, ..., x_{\ell-1}]$ denotes concatenation of all preceding feature maps and $H_\ell$ represents a composite function of batch normalization, ReLU, and convolution.

This dense connectivity encourages feature reuse, improves gradient flow, and reduces parameters through aggressive feature map reduction. DenseNet-121 contains 121 layers organized into four dense blocks, totaling 8 million parameters. The architecture typically achieves competitive accuracy with substantially fewer parameters than ResNet variants. I again employed ImageNet pretraining with final layer adaptation.

### 3.2.4 Vision Transformer (ViT-B/16)

Vision Transformers (Dosovitskiy et al., 2021) adapt the transformer architecture from NLP to vision by treating images as sequences of patches. ViT-B/16 divides input images into 16×16 pixel patches (producing 14×14 = 196 patches for 224×224 images), projects each patch to a learned embedding, adds positional encodings, and processes the sequence through 12 transformer encoder layers.

Unlike CNNs which build translation equivariance through convolution, transformers learn spatial relationships through attention mechanisms:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where query (Q), key (K), and value (V) matrices derive from learned projections of patch embeddings.

ViT-B/16 contains 86 million parameters, substantially more than the CNN architectures. While transformers often require large-scale pretraining to achieve peak performance, ImageNet pretraining provides a reasonable initialization. I maintained the patch projection and transformer layers unchanged, only adapting the final classification head to 7 classes.

## 3.3 Training Procedure

### 3.3.1 Hyperparameters

All models were trained with identical hyperparameters to ensure fair comparison:

- **Optimizer**: Adam with $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$
- **Learning Rate**: $1 \times 10^{-4}$ (fixed, no scheduling)
- **Batch Size**: 32 images per batch
- **Epochs**: 20 epochs (full passes through training data)
- **Loss Function**: Cross-entropy loss
- **Weight Decay**: No explicit L2 regularization (implicit regularization through dropout in architectures)
- **Random Seed**: 42 (for reproducibility)

The learning rate was selected based on preliminary experiments and represents a conservative choice that provides stable training across architectures. While adaptive learning rate schedules or larger initial rates followed by decay often improve final performance, using a fixed learning rate simplifies comparison by removing an additional hyperparameter that might favor some architectures over others.

### 3.3.2 Implementation Details

Training was implemented in PyTorch 2.8.0 using CUDA for GPU acceleration. Models were trained on an NVIDIA GPU with automatic mixed precision (AMP) to improve training speed and reduce memory consumption. Batch normalization layers used running statistics accumulated during training and switched to evaluation mode during validation.

For each architecture, training proceeded as follows:

1. **Initialization**: Load ImageNet pretrained weights and replace final classification layer
2. **Training Loop**: For each epoch:
   - Shuffle training data
   - Iterate through training batches with gradient accumulation
   - Apply data augmentation on-the-fly
   - Compute forward pass, loss, and gradients
   - Update parameters using Adam optimizer
   - Evaluate on validation set (no augmentation)
   - Record training loss, training accuracy, validation loss, validation accuracy
3. **Checkpoint Saving**: Save model state after each epoch, keeping the checkpoint with best validation accuracy

### 3.3.3 Training Time

Total training time for all four models (4 models × 20 epochs = 80 epochs) was approximately 2 hours on a single GPU. Individual epoch times varied by architecture:

- ResNet-50: ~2-3 minutes per epoch
- EfficientNet-B3: ~3-4 minutes per epoch  
- DenseNet-121: ~3-4 minutes per epoch
- ViT-B/16: ~2-3 minutes per epoch

The relatively short training time stems from moderate dataset size (8,511 training images), efficient data loading with prefetching, and optimized implementation. Transfer learning from ImageNet pretrained weights enables effective training with fewer epochs than training from scratch would require.

## 3.4 Temperature Calibration

### 3.4.1 Motivation

Standard neural network training optimizes classification accuracy through cross-entropy loss, which encourages correct predictions but does not explicitly optimize probability calibration. Modern deep networks, particularly with extensive regularization or trained for many epochs, often produce overconfident predictions where the maximum softmax probability systematically exceeds empirical accuracy.

For clinical applications, calibrated probabilities are essential: a model that assigns 90% confidence should be correct approximately 90% of the time. Miscalibration can mislead clinical decision-making, with overconfidence particularly dangerous if it leads to inappropriate treatment of false positive predictions or dismissal of false negative warnings.

### 3.4.2 Temperature Scaling Method

Following Guo et al. (2017), I applied temperature scaling as a post-processing calibration method. The technique introduces a single scalar parameter $T$ that divides logits before softmax:

$$\hat{P}(y=k|x) = \frac{\exp(z_k/T)}{\sum_j \exp(z_j/T)}$$

where $z_k$ represents the model's logit (pre-softmax output) for class $k$, and $\hat{P}$ denotes calibrated probabilities.

Temperature $T > 1$ softens the probability distribution by reducing the relative differences between logits, decreasing maximum probability and improving calibration for overconfident models. Temperature $T < 1$ would sharpen distributions, though overconfidence is more common than underconfidence in modern networks.

Critically, temperature scaling maintains the predicted class (argmax of probabilities) since dividing all logits by the same value preserves their relative ordering. This means calibration improves probability estimates without affecting accuracy, specificity, or other rank-based metrics.

### 3.4.3 Optimization Procedure

For each trained model, I optimized temperature $T$ on the validation set:

1. **Extract Logits**: Run the trained model on all validation images, storing logits (pre-softmax outputs)
2. **Temperature Search**: Minimize negative log-likelihood with respect to $T$:
   $$T^* = \argmin_T -\sum_{i=1}^N \log \hat{P}(y=y_i|x_i, T)$$
   where $N$ is validation set size and $y_i$ are true labels
3. **Grid Search**: I used scipy's optimization to find $T^*$ over the range [0.1, 10.0]

The optimization typically requires only seconds as it involves a single pass through the validation set and optimization over a single scalar parameter.

### 3.4.4 Calibration Metrics

I evaluated calibration quality using Expected Calibration Error (ECE) and reliability diagrams:

**Expected Calibration Error**: Partition predictions into $M=15$ bins based on confidence (maximum softmax probability). Within each bin $B_m$, compute accuracy $\text{acc}(B_m)$ and average confidence $\text{conf}(B_m)$. ECE is the weighted average of absolute differences:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$

Well-calibrated models show ECE close to 0, indicating confidence matches accuracy across all confidence levels.

**Brier Score**: Measures mean squared error between predicted probabilities and true labels (one-hot encoded):

$$\text{BS} = \frac{1}{N}\sum_{i=1}^N \sum_{k=1}^K (p_{ik} - y_{ik})^2$$

where $p_{ik}$ is predicted probability for class $k$ and $y_{ik}$ is 1 if true class is $k$, else 0. Lower Brier scores indicate better probability estimates combining both calibration and accuracy.

## 3.5 Operating Threshold Selection

### 3.5.1 Clinical Context

Binary classification typically uses a decision threshold of 0.5: predict positive if $P(y=\text{positive}) > 0.5$. However, this threshold implicitly assumes equal costs for false positives and false negatives, which rarely holds in medical contexts.

For melanoma screening, false negatives (missing melanoma) carry serious consequences including disease progression and reduced survival. False positives (unnecessary biopsies of benign lesions) incur costs including patient anxiety, procedure risks, and healthcare expenses, but these are generally less severe than missing malignant lesions.

However, in a screening context with low disease prevalence (11% melanoma in HAM10000), even modest false positive rates can overwhelm dermatology practices with excessive referrals. A system with 95% specificity still produces 5 false positives per 100 benign lesions, meaning with 67% benign nevi prevalence, we would see approximately 3.4 false positives per true positive at baseline prevalence.

### 3.5.2 Specificity-Constrained Threshold

I established operating thresholds targeting 95% specificity for melanoma detection. This choice reflects a balance: high specificity controls false positive rate, while achieved sensitivity determines the system's utility as a screening tool.

For each model, I:

1. **Extract Melanoma Probabilities**: Compute calibrated probability of melanoma for all validation images
2. **Compute ROC Curve**: Vary threshold and calculate sensitivity and specificity at each point
3. **Find 95% Specificity Threshold**: Select threshold where specificity equals or exceeds 95%
4. **Report Operating Point Metrics**: At the selected threshold, report:
   - Sensitivity (true positive rate)
   - Specificity (true negative rate)
   - Positive Predictive Value (precision)
   - Negative Predictive Value

This threshold becomes the recommended decision boundary when deploying the model for screening: lesions with calibrated melanoma probability above the threshold warrant further examination or biopsy, while those below may be monitored.

## 3.6 Explainability Implementation

### 3.6.1 Grad-CAM Algorithm

I implemented Gradient-weighted Class Activation Mapping (Grad-CAM) to generate attention heatmaps highlighting image regions most influential to model predictions. The algorithm operates as follows:

1. **Forward Pass**: Process input image through the network, storing activations $A^k$ from the final convolutional layer (where $k$ indexes feature maps)

2. **Backward Pass**: Compute gradients of the class score $y^c$ with respect to feature map activations:
   $$\frac{\partial y^c}{\partial A^k_{ij}}$$

3. **Global Average Pooling**: Average gradients spatially to obtain importance weights:
   $$\alpha_k^c = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A^k_{ij}}$$
   where $Z$ is the number of spatial locations

4. **Weighted Combination**: Combine feature maps using importance weights:
   $$L_{Grad-CAM}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

5. **Upsampling**: Resize the resulting heatmap to input image resolution using bilinear interpolation

The ReLU activation captures only features with positive influence on the target class (pixels that increase the class score), filtering out negative influences that would create confusing visualizations.

### 3.6.2 Implementation Details

For each architecture, I identified the final convolutional layer:

- **ResNet-50**: layer4[2].conv3 (final conv in last residual block)
- **EfficientNet-B3**: features[8] (final efficient block)
- **DenseNet-121**: features.denseblock4 (final dense block)
- **ViT-B/16**: Attention maps from final encoder layer

Implementation used PyTorch hooks to capture activations and gradients during forward/backward passes without modifying the trained models. For each prediction, I generated Grad-CAM heatmaps for the predicted class and overlaid them on the original images using a jet colormap with 40% transparency.

### 3.6.3 Validation Approach

I qualitatively validated Grad-CAM outputs by examining attention maps for correctly classified melanoma and nevi cases, verifying that:

1. **Lesion Focus**: Attention primarily concentrates on the lesion rather than background skin or image artifacts
2. **ABCDE Correlation**: For melanoma, attention often highlights regions exhibiting ABCDE criteria (asymmetric regions, irregular borders, color variation)
3. **Consistency**: Similar lesions produce similar attention patterns across different images
4. **Failure Analysis**: For misclassifications, attention maps sometimes reveal plausible reasoning (ambiguous features) rather than obviously spurious correlations

While fully quantitative validation of explainability remains challenging, these qualitative checks provide confidence that models learn medically meaningful representations rather than exploiting dataset artifacts.

## 3.7 Web Interface Development

### 3.7.1 Gradio Framework

I developed an interactive web interface using Gradio 5.49.1, a Python library for rapidly building machine learning demos. The interface allows users to:

1. Upload dermoscopic images
2. Receive predicted probabilities for all seven lesion categories
3. View Grad-CAM attention heatmaps
4. Read AI-generated explanations contextualizing predictions
5. Ask follow-up questions through an interactive chat interface

### 3.7.2 Interface Components

**Image Upload**: Users drag-and-drop or browse for dermoscopic images in common formats (JPEG, PNG). Uploaded images are automatically resized and normalized to match training preprocessing.

**Prediction Display**: Results show calibrated probabilities for all seven diagnostic categories, displayed both numerically (percentage) and as a horizontal bar chart. The predicted class (highest probability) is highlighted.

**Grad-CAM Overlay**: An attention heatmap overlays the original image, using a jet colormap (blue=low attention, red=high attention) with transparency to preserve underlying image details.

**AI Explanation**: A text explanation analyzes the prediction, mentioning:
- Predicted diagnosis with confidence level
- Relevant ABCDE criteria observed (for melanoma or concerning lesions)
- Attention map interpretation (which regions influenced the decision)
- Clinical recommendation (e.g., "Further examination recommended" for melanoma)

**Interactive Chat**: Users can ask questions about the diagnosis, and an AI assistant provides contextual responses based on the prediction and general dermatological knowledge.

### 3.7.3 Disclaimer and Limitations

The interface prominently displays medical disclaimers:

> "This system is for research purposes only and does not constitute medical advice. Consult a qualified dermatologist for proper diagnosis and treatment. AI predictions should be used as decision support, not autonomous diagnosis."

These disclaimers remind users that the system serves as a screening and education tool, not a replacement for professional medical evaluation.

## 3.8 Evaluation Metrics

Beyond calibration and operating threshold metrics, I evaluated models using standard classification metrics:

**Accuracy**: Fraction of correct predictions across all classes
$$\text{Accuracy} = \frac{\text{TP + TN}}{\text{Total}}$$

**AUC (Area Under ROC Curve)**: Measures discriminative ability across all possible thresholds. Macro-averaged AUC treats all classes equally, while melanoma-specific AUC focuses on the primary clinical task.

**Confusion Matrix**: Shows prediction distribution across true and predicted classes, revealing which categories are commonly confused.

**Inference Time**: Average time to process a single image, including preprocessing, forward pass, and post-processing. Measured over 100 runs with median reported to reduce noise from system variation.

**Sensitivity and Specificity**: At the chosen operating threshold for melanoma:
- Sensitivity = TP / (TP + FN) (true positive rate)
- Specificity = TN / (TN + FP) (true negative rate)

**Positive and Negative Predictive Values**: At the operating threshold:
- PPV = TP / (TP + FP) (precision)
- NPV = TN / (TN + FN)

These metrics provide a comprehensive view of model performance from multiple clinically-relevant perspectives.
