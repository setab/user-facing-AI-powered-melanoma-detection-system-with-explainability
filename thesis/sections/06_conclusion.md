# 6. Conclusion

## 6.1 Summary of Work

This thesis addressed the challenge of developing a practically deployable melanoma detection system by integrating classification accuracy, probability calibration, explainability, and accessible deployment into a coherent framework. I systematically compared four contemporary deep learning architectures—ResNet-50, EfficientNet-B3, DenseNet-121, and Vision Transformer—on the HAM10000 dataset of 10,013 dermoscopic images spanning seven diagnostic categories.

EfficientNet-B3 emerged as the best-performing architecture, achieving 89.22% overall accuracy and 95.34% AUC for melanoma detection. At a clinically-motivated operating point (95% specificity), the model demonstrated 80.72% sensitivity for melanoma identification, approaching dermatologist-level performance. Temperature scaling improved probability calibration across all models, reducing expected calibration error from 8-9% to approximately 2.7%, enabling reliable interpretation of predicted probabilities for clinical decision-making.

Beyond classification metrics, the system provides visual explanations through Gradient-weighted Class Activation Mapping, highlighting dermoscopic regions that influence model predictions. Qualitative validation showed these attention maps correlate with established ABCDE melanoma criteria, indicating models learn diagnostically meaningful representations. An interactive web interface built with Gradio packages the complete system—prediction, calibration, visualization, explanation—into an accessible tool that healthcare providers can use with minimal technical expertise.

## 6.2 Key Contributions

This work makes several contributions to the intersection of deep learning and medical imaging:

### 6.2.1 Systematic Multi-Architecture Comparison

By training ResNet-50, EfficientNet-B3, DenseNet-121, and ViT-B/16 under identical conditions—same dataset split, preprocessing, hyperparameters, and training duration—this work provides fair comparison attributing performance differences to architectural choices rather than experimental variations. The finding that EfficientNet-B3 achieves superior accuracy with fewer parameters than larger models demonstrates the value of neural architecture search over naive parameter scaling.

### 6.2.2 Integrated Clinical Calibration Framework

The combination of temperature scaling for probability calibration with specificity-constrained threshold optimization demonstrates a general methodology for deploying deep learning in medical screening contexts. This framework addresses the gap between machine learning performance metrics (accuracy, AUC) and clinical decision requirements (operating points balancing sensitivity and specificity based on real-world costs and benefits). The approach generalizes beyond melanoma detection to other medical screening applications.

### 6.2.3 Validated Explainable AI

Rather than treating explainability as an add-on, the system design incorporates Grad-CAM attention visualization as a core feature and validates that attention maps correlate with dermatological diagnostic criteria (ABCDE features). This validation demonstrates that convolutional neural networks can learn medically interpretable feature representations when trained on appropriate data, addressing concerns about black-box models in high-stakes medical decisions.

### 6.2.4 Complete Deployable System

Many research systems remain proof-of-concept demonstrations. This work delivers a complete, documented implementation including data pipelines, trained models with calibration parameters, explainability components, and an interactive web interface ready for clinical pilot deployment. All code, model configurations, and procedures are documented in detail, enabling reproduction and adaptation to related medical imaging domains.

## 6.3 Impact and Significance

### 6.3.1 Clinical Utility

The achieved performance—95.34% melanoma AUC, 80.72% sensitivity at 95% specificity—places this system in the range of expert dermatologist capability. While direct clinical deployment requires additional validation, the system demonstrates the feasibility of AI-assisted melanoma screening that could:

- **Extend expertise** to primary care settings and underserved populations lacking dermatologist access
- **Provide second opinions** helping dermatologists verify diagnoses for challenging cases
- **Triage referrals** by prioritizing high-risk lesions for specialist evaluation
- **Enable self-screening** through consumer applications (though this requires careful design to avoid misuse)

### 6.3.2 Methodological Advances

The calibration framework and validated explainability methodology provide templates applicable beyond melanoma detection. Medical AI systems require not just accuracy but calibrated probabilities, operating thresholds matched to clinical requirements, and explanations that build clinical trust. This work demonstrates how to integrate these components systematically rather than treating them as afterthoughts.

### 6.3.3 Research Foundation

By providing trained models, detailed documentation, and reproducible methodology on a public dataset, this work establishes a foundation for future research. Others can extend the approach through improved architectures, enhanced training techniques, or adaptation to related dermoscopic diagnosis tasks (basal cell carcinoma detection, psoriasis severity assessment) without replicating basic infrastructure.

## 6.4 Limitations and Future Work

Despite promising results, several limitations warrant acknowledgment and suggest directions for future research:

### 6.4.1 Dataset Constraints

The HAM10000 dataset, while substantial for medical imaging, remains moderate-sized by computer vision standards. Geographic concentration in Austria and Australia potentially biases toward lighter skin tones, raising equity concerns about performance on diverse populations. Class imbalance led to poor performance on rare categories (vascular lesions, dermatofibroma), suggesting need for class balancing techniques or larger datasets.

Future work should validate models on diverse populations, incorporate multi-institutional data, and explore data augmentation or synthetic data generation to address class imbalance and improve generalization.

### 6.4.2 Single-Image Limitation

Clinical melanoma diagnosis often incorporates lesion evolution over time—a key ABCDE criterion this work cannot capture from single images. Longitudinal data showing lesion changes could substantially improve detection accuracy and reduce false positives on stable benign lesions that superficially resemble melanoma.

Developing models that process temporal sequences of dermoscopic images represents an important research direction. Recurrent neural networks, temporal convolutions, or video transformers could enable tracking lesion evolution and incorporating change patterns into predictions.

### 6.4.3 Explainability Depth

While Grad-CAM provides useful attention visualization, the explanations remain coarse (limited by convolutional layer resolution) and post-hoc (generated after training rather than intrinsic to model design). More sophisticated attribution methods (integrated gradients, concept-based explanations) or inherently interpretable architectures might provide richer, more faithful explanations that better support clinical reasoning.

### 6.4.4 Clinical Validation Gap

This work demonstrates technical feasibility but lacks clinical deployment validation. Prospective studies measuring real-world diagnostic impact—whether AI assistance improves dermatologist accuracy, reduces time-to-diagnosis, or affects patient outcomes—provide the strongest evidence of utility. Cost-effectiveness analyses and integration studies would inform adoption decisions beyond technical performance metrics.

## 6.5 Broader Perspective

This thesis contributes to the growing body of evidence that deep learning can achieve expert-level performance on well-defined medical imaging tasks. The findings support cautious optimism about AI's potential to augment clinical capabilities, particularly in addressing access disparities and supporting non-specialist providers.

However, the limitations and challenges discussed underscore that technical performance alone does not ensure clinical value. Successful deployment requires addressing workflow integration, regulatory approval, clinician trust, patient acceptance, and ethical considerations around bias and accountability. Medical AI development is not purely a technical problem but a sociotechnical challenge requiring collaboration among computer scientists, clinicians, ethicists, and policymakers.

The open-source ethos of this work—public data, reproducible methodology, shared code—reflects the belief that medical AI progress benefits from community collaboration rather than proprietary competition. As the field matures from research demonstrations toward clinical deployment, maintaining transparency, reproducibility, and accessibility will be essential for realizing AI's potential to improve healthcare equitably.

## 6.6 Concluding Remarks

Melanoma detection from dermoscopic images represents a well-suited application for deep learning: the diagnostic task fundamentally involves visual pattern recognition, large public datasets exist for training, and clinical need for decision support is clear given diagnostic difficulty and dermatologist shortages. This work demonstrates that modern convolutional neural networks, when properly trained, calibrated, and deployed, can achieve performance approaching human experts while providing explainable predictions.

The integrated system developed here—combining EfficientNet-B3 classification, temperature-scaled probability calibration, specificity-constrained thresholds, Grad-CAM visualization, and an accessible web interface—illustrates how research prototypes can evolve toward clinically-deployable tools. While substantial work remains before broad clinical adoption, the results provide evidence that AI-assisted melanoma detection can meaningfully contribute to early diagnosis and improved patient outcomes.

Future research should focus on enhancing generalization through multi-institutional validation, incorporating temporal information through longitudinal imaging, deepening explainability through more sophisticated attribution methods, and conducting prospective clinical trials to measure real-world impact. Additionally, addressing ethical concerns around bias, privacy, and equitable access must parallel technical development to ensure AI benefits all populations fairly.

Ultimately, the goal is not AI replacing dermatologists but augmenting clinical capabilities—extending expertise to underserved populations, reducing diagnostic errors, and freeing specialists to focus on complex cases requiring human judgment. This thesis takes a step toward that vision by demonstrating that integrated, explainable, calibrated AI systems for melanoma detection are technically feasible and, with appropriate validation and safeguards, could provide meaningful clinical value.
