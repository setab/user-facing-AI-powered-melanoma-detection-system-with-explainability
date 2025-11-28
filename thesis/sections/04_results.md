# 4. Results

## 4.1 Overall Performance Comparison

Table 4.1 summarizes classification performance across the four architectures. All models achieved validation accuracies exceeding 84%, demonstrating that transfer learning from ImageNet enables effective dermoscopic image classification despite domain differences. EfficientNet-B3 achieved the highest accuracy at 89.22%, followed by ResNet-50 (87.27%), DenseNet-121 (86.12%), and ViT-B/16 (84.47%).

**Table 4.1: Model Performance Summary**

| Architecture | Accuracy | Macro AUC | ECE (%) | Brier Score | Inference (ms) | Parameters |
|--------------|----------|-----------|---------|-------------|----------------|------------|
| ResNet-50 | 87.27% | 97.62% | 2.73 | 0.0266 | 4.89 ± 0.82 | 25M |
| **EfficientNet-B3** | **89.22%** | **97.02%** | **2.71** | **0.0234** | 10.06 ± 1.36 | 12M |
| DenseNet-121 | 86.12% | 97.28% | 2.73 | 0.0286 | 12.01 ± 1.40 | 8M |
| ViT-B/16 | 84.47% | 96.72% | 2.74 | 0.0328 | 3.58 ± 0.76 | 86M |

These results reveal several interesting patterns:

**Parameter Efficiency**: EfficientNet-B3 achieved the best accuracy with only 12 million parameters, demonstrating the effectiveness of neural architecture search and compound scaling. ResNet-50, with twice the parameters, achieved 2% lower accuracy. ViT-B/16, despite having 86 million parameters (7× more than EfficientNet), showed the lowest accuracy, suggesting that transformer architectures may require larger datasets or extended pretraining to reach their full potential on this task.

**Speed-Accuracy Tradeoff**: ViT-B/16 offered the fastest inference at 3.58ms per image, making it attractive for high-throughput applications despite lower accuracy. DenseNet-121, with the fewest parameters (8M), showed the slowest inference (12.01ms), likely due to the computational overhead of concatenating feature maps in dense blocks. EfficientNet-B3's 10.06ms inference time represents a reasonable middle ground given its superior accuracy.

**Calibration Quality**: All models achieved expected calibration errors below 3% after temperature scaling, indicating well-calibrated probability estimates. EfficientNet-B3 showed the best calibration (2.71% ECE) alongside the lowest Brier score (0.0234), suggesting its probability estimates most accurately reflect true diagnostic uncertainty.

## 4.2 Training Dynamics

Figure 4.1 shows training and validation learning curves for all architectures over 20 epochs. Several patterns emerge:

**Convergence Behavior**: All models showed rapid initial improvement in the first 5 epochs, with validation accuracy increasing from ~75% to 82-85%. After epoch 10, learning curves plateaued, with validation accuracy stabilizing while training accuracy continued to improve slightly, indicating mild overfitting despite data augmentation.

**EfficientNet-B3 Progression**: EfficientNet-B3 demonstrated the steadiest improvement, reaching 89.22% validation accuracy by epoch 17 and maintaining this level. The model showed minimal overfitting, with training accuracy (97.3%) remaining reasonably close to validation accuracy (89.2%).

**ResNet-50 Stability**: ResNet-50 achieved 87.3% validation accuracy by epoch 14 with relatively smooth training curves and minimal noise, reflecting the architecture's maturity and stability.

**DenseNet-121 Variability**: DenseNet-121 showed more variable validation performance, with accuracy fluctuating between 85% and 87% after epoch 10 before stabilizing at 86.1%. This variability may reflect the architecture's sensitivity to batch composition or learning rate.

**ViT-B/16 Slower Learning**: ViT-B/16 showed slower initial learning, reaching only 78% accuracy by epoch 5 (compared to 82-83% for CNNs). The model continued improving steadily through epoch 20, suggesting that additional training epochs might have further improved performance. This slower learning aligns with observations that transformers often require more training than CNNs.

## 4.3 Melanoma-Specific Performance

Given that melanoma detection represents the primary clinical motivation, Table 4.2 presents melanoma-specific metrics at operating thresholds chosen for 95% specificity.

**Table 4.2: Melanoma Detection Performance**

| Architecture | AUC | Threshold | Sensitivity | Specificity | PPV | NPV |
|--------------|-----|-----------|-------------|-------------|-----|-----|
| ResNet-50 | 94.25% | 0.1721 | 68.61% | 95.00% | 63.22% | 96.02% |
| **EfficientNet-B3** | **95.34%** | 0.2191 | **80.72%** | 95.17% | 67.67% | 97.52% |
| DenseNet-121 | 93.76% | 0.2535 | 65.02% | 95.00% | 61.97% | 95.59% |
| ViT-B/16 | 92.44% | 0.5394 | 60.54% | 94.78% | 59.21% | 95.04% |

**AUC Analysis**: All models achieved melanoma AUC above 92%, indicating strong discriminative ability. EfficientNet-B3's 95.34% AUC exceeded the next-best architecture (ResNet-50 at 94.25%) by over 1 percentage point. These AUC values approach or match dermatologist performance reported in literature (typically 90-95% for dermoscopic melanoma detection).

**Sensitivity at 95% Specificity**: EfficientNet-B3 achieved 80.72% sensitivity while maintaining 95% specificity, meaning it correctly identified approximately 4 out of 5 melanomas while producing false positives on only 5% of benign lesions. This represents the best sensitivity among all evaluated models at this operating point.

ResNet-50 achieved 68.61% sensitivity—about 12 percentage points lower than EfficientNet-B3. DenseNet-121 (65.02%) and ViT-B/16 (60.54%) showed even lower sensitivity at the 95% specificity constraint. These differences translate to substantial clinical impact: at 80.72% sensitivity, EfficientNet-B3 would catch 135 of 167 melanomas in the validation set, while ViT-B/16 at 60.54% would catch only 101—a difference of 34 missed melanomas.

**Predictive Values**: Negative predictive value (NPV) exceeded 95% for all models, reflecting the relatively high prevalence of benign lesions. A negative prediction (melanoma probability below threshold) strongly indicates benign disease. Positive predictive value (PPV) ranged from 59% to 68%, meaning that roughly 1 in 3 positive predictions were false alarms—still clinically useful as these would prompt biopsy, the definitive diagnostic procedure.

**Threshold Variation**: The optimal threshold varied substantially across models, from 0.1721 (ResNet-50) to 0.5394 (ViT-B/16). This reflects different confidence levels in model predictions: ViT-B/16 required high confidence (>53.9%) to achieve 95% specificity, while ResNet-50 achieved similar specificity with much lower threshold (17.2%). This illustrates why operating threshold optimization is essential rather than using arbitrary cutoffs like 0.5.

## 4.4 Confusion Matrix Analysis

Figure 4.2 shows confusion matrices for all models. Several patterns of diagnostic confusion emerge:

**Melanocytic Nevi vs. Melanoma**: The most clinically significant confusion occurs between melanocytic nevi (benign moles) and melanoma. EfficientNet-B3 misclassified only 12 melanomas as nevi (7.2% of melanomas) and 26 nevi as melanoma (2.6% of nevi). In contrast, ViT-B/16 misclassified 25 melanomas as nevi (15.0%) and showed more variable confusion patterns. This distinction directly impacts clinical utility, as melanoma-nevi confusion carries the highest risk.

**Benign Keratosis Confusion**: Benign keratosis-like lesions showed confusion with melanocytic nevi across all models, which is clinically acceptable as both are benign. More concerning, some keratoses were misclassified as melanoma (16 cases for EfficientNet-B3), potentially leading to unnecessary biopsies.

**Actinic Keratoses Misclassification**: Actinic keratoses, with only 49 validation samples, showed high misclassification rates. EfficientNet-B3 misclassified 26 of 49 as melanocytic nevi and 7 as basal cell carcinoma. This reflects both the small sample size and genuine diagnostic difficulty, as actinic keratoses can show variable dermoscopic features.

**Rare Classes**: Vascular lesions (21 validation samples) and dermatofibroma (17 samples) showed substantial misclassification, with many predicted as melanocytic nevi. This indicates that the models default to common classes for rare or ambiguous cases—a manifestation of the class imbalance problem.

**Multi-class Accuracy**: Beyond melanoma detection, overall classification across all seven categories achieved 89.22% accuracy for EfficientNet-B3. While lower than binary melanoma vs. non-melanoma classification would achieve, multi-class prediction provides more diagnostic information and better reflects the clinical decision space.

## 4.5 Temperature Calibration Results

Figure 4.3 shows reliability diagrams before and after temperature scaling. Pre-calibration, all models showed overconfidence, with predicted probabilities systematically exceeding empirical accuracy, particularly at high confidence levels. Post-calibration, probabilities aligned closely with the diagonal, indicating well-calibrated predictions.

**Optimal Temperatures**:
- ResNet-50: T = 1.616
- EfficientNet-B3: T = 1.647
- DenseNet-121: T = 1.627
- ViT-B/16: T = 1.625

All optimal temperatures exceeded 1, confirming systematic overconfidence in raw model outputs. The temperatures cluster tightly around 1.62, suggesting this calibration behavior may be characteristic of ImageNet-pretrained models fine-tuned on moderate-sized medical datasets.

**Calibration Improvement**: Expected Calibration Error decreased substantially with temperature scaling:

| Architecture | ECE (raw) | ECE (calibrated) | Improvement |
|--------------|-----------|------------------|-------------|
| ResNet-50 | 8.42% | 2.73% | -67.6% |
| EfficientNet-B3 | 7.89% | 2.71% | -65.7% |
| DenseNet-121 | 8.91% | 2.73% | -69.4% |
| ViT-B/16 | 9.24% | 2.74% | -70.3% |

Raw models showed ECE between 7.9% and 9.2%, indicating substantial miscalibration. Temperature scaling reduced ECE to 2.7% for all models, achieving roughly 65-70% improvement. The remaining 2.7% error reflects irreducible calibration difficulty: even with perfect temperature, some binning error and finite-sample effects persist.

**Brier Score**: Brier scores improved modestly with calibration, as they combine discrimination (accuracy) and calibration. EfficientNet-B3 showed the best Brier score (0.0234), followed by ResNet-50 (0.0266), DenseNet-121 (0.0286), and ViT-B/16 (0.0328).

## 4.6 Grad-CAM Visualization Analysis

Figures 4.4 and 4.5 show representative Grad-CAM attention maps for correctly classified melanoma and melanocytic nevi cases.

**Melanoma Attention Patterns**: For melanoma cases, Grad-CAM consistently highlighted regions exhibiting ABCDE criteria:
- **Asymmetry**: Attention concentrated on irregular growth regions where one half differed from the other
- **Border Irregularity**: High attention along jagged, poorly-defined borders
- **Color Variation**: Multiple attention peaks corresponding to regions with dark pigmentation, red inflammation, or blue-gray regression

These patterns align with dermatological knowledge, suggesting models learn clinically meaningful features rather than spurious correlations. In several cases, the attention maps highlighted subtle features (small irregular regions, minor color variations) that might challenge less experienced clinicians, demonstrating the model's pattern recognition capability.

**Melanocytic Nevi Patterns**: For benign nevi, attention typically distributed more evenly across the lesion or focused on characteristic symmetric patterns. Heatmaps showed less pronounced peaks and more diffuse activation, reflecting the lack of highly discriminative irregular features. In some cases, attention highlighted pigment networks—regular, mesh-like patterns characteristic of benign nevi.

**Failure Case Analysis**: For melanomas misclassified as benign, attention maps often revealed plausible explanations. Some melanomas exhibited relatively regular features (symmetric shape, uniform color) that justified lower malignancy predictions. Conversely, benign lesions misclassified as melanoma often showed ambiguous features (slight asymmetry, multiple colors) that could reasonably raise concern. These failure cases suggest model errors often reflect genuine diagnostic ambiguity rather than obvious mistakes.

**Cross-Architecture Consistency**: Attention maps showed reasonable consistency across architectures for the same images. Different models often highlighted similar regions, though with varying spatial precision (ViT-B/16 showed coarser attention due to patch-based processing). This consistency suggests the architectures learn related feature representations despite different design philosophies.

## 4.7 Inference Time Analysis

Figure 4.6 compares inference times across architectures. ViT-B/16 achieved the fastest inference (3.58 ± 0.76 ms), followed by ResNet-50 (4.89 ± 0.82 ms), EfficientNet-B3 (10.06 ± 1.36 ms), and DenseNet-121 (12.01 ± 1.40 ms).

All inference times remain well below human decision times (typically seconds to minutes for dermatologists), making any of these models suitable for real-time clinical use. The 2-3× speed advantage of ViT-B/16 over EfficientNet-B3 might matter for high-throughput batch processing (screening thousands of images) but provides minimal benefit in interactive clinical use where image acquisition time dominates.

The relatively slow inference of DenseNet-121 despite having the fewest parameters (8M) likely reflects the computational cost of concatenating feature maps in dense blocks. EfficientNet-B3's 10ms inference represents a reasonable tradeoff given its superior accuracy.

## 4.8 Statistical Significance

I performed McNemar's test to assess whether accuracy differences between models are statistically significant. Comparing EfficientNet-B3 (best) to other architectures on the 1,502 validation images:

- **EfficientNet-B3 vs. ResNet-50**: p = 0.041 (significant at α = 0.05)
- **EfficientNet-B3 vs. DenseNet-121**: p = 0.003 (highly significant)
- **EfficientNet-B3 vs. ViT-B/16**: p < 0.001 (highly significant)

These results confirm that EfficientNet-B3's accuracy improvement is statistically significant rather than random variation. The 2% accuracy advantage over ResNet-50 and 5% advantage over ViT-B/16 reflect genuine performance differences.

## 4.9 Summary of Key Findings

The experimental results support several conclusions:

1. **EfficientNet-B3 Superiority**: Across accuracy, calibration, melanoma AUC, and sensitivity at 95% specificity, EfficientNet-B3 consistently outperformed other architectures. The combination of neural architecture search and compound scaling proved highly effective for this task.

2. **All Models Clinically Viable**: Even the lowest-performing architecture (ViT-B/16) achieved 84.47% accuracy and 92.44% melanoma AUC, approaching dermatologist-level performance. All models provide clinically useful decision support when deployed with appropriate thresholds and calibration.

3. **Calibration Essential**: Raw model outputs showed substantial overconfidence (8-9% ECE), which temperature scaling reduced to ~2.7%. This calibration enables reliable interpretation of predicted probabilities for clinical decision-making.

4. **Explainability Validation**: Grad-CAM attention maps correlated with established ABCDE criteria, indicating models learn diagnostically meaningful representations. This explainability facilitates clinical trust and enables error analysis.

5. **Class Imbalance Challenge**: Rare categories (vascular lesions, dermatofibroma) showed high misclassification rates, suggesting that future work should address class imbalance through techniques like oversampling, class weighting, or focal loss.

6. **Parameter Efficiency**: Model size and parameter count did not directly predict performance. EfficientNet-B3 (12M parameters) outperformed ResNet-50 (25M) and ViT-B/16 (86M), highlighting the importance of architecture design over scale alone.
