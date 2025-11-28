# Deep Learning for Melanoma Detection from Dermoscopic Images: A Systematic Comparison of Architectures with Calibrated Probabilities and Explainable Predictions

**A Thesis Presented for Research in Medical Artificial Intelligence**

---

## Author Information

**Author**: [Your Name]  
**Institution**: [Your Institution]  
**Department**: [Your Department]  
**Date**: November 2025

---

## Abstract

[See sections/00_abstract.md for full abstract]

Early detection remains critical in melanoma diagnosis, where treatment success strongly correlates with disease stage at identification. This work addresses the need for accessible, reliable diagnostic support by developing an integrated deep learning system that combines accurate classification with clinically-relevant explainability features.

I implemented and compared four convolutional neural network architectures on the HAM10000 dataset containing 10,013 dermoscopic images. EfficientNet-B3 emerged as the best-performing architecture, achieving 89.22% overall accuracy and 95.34% AUC for melanoma detection, with 80.72% sensitivity at 95% specificity.

**Key Results**:
- Best Model: EfficientNet-B3 (89.22% accuracy, 95.34% melanoma AUC)
- All models achieved <3% Expected Calibration Error after temperature scaling
- Grad-CAM attention maps correlated with clinical ABCDE criteria
- Real-time inference (<12ms per image) suitable for clinical deployment
- Interactive web interface deployed for practical use

---

## Table of Contents

1. [Introduction](sections/01_introduction.md)
   - 1.1 Background and Motivation
   - 1.2 Problem Statement
   - 1.3 Approach Overview
   - 1.4 Key Contributions
   - 1.5 Thesis Organization

2. [Background and Related Work](sections/02_background.md)
   - 2.1 Clinical Context: Melanoma Diagnosis
   - 2.2 Deep Learning for Medical Imaging
   - 2.3 Deep Learning in Dermatology
   - 2.4 Model Calibration
   - 2.5 Explainable AI in Medical Imaging
   - 2.6 Web-Based Medical AI Systems

3. [Methodology](sections/03_methodology.md)
   - 3.1 Dataset
   - 3.2 Model Architectures
   - 3.3 Training Procedure
   - 3.4 Temperature Calibration
   - 3.5 Operating Threshold Selection
   - 3.6 Explainability Implementation
   - 3.7 Web Interface Development
   - 3.8 Evaluation Metrics

4. [Results](sections/04_results.md)
   - 4.1 Overall Performance Comparison
   - 4.2 Training Dynamics
   - 4.3 Melanoma-Specific Performance
   - 4.4 Confusion Matrix Analysis
   - 4.5 Temperature Calibration Results
   - 4.6 Grad-CAM Visualization Analysis
   - 4.7 Inference Time Analysis
   - 4.8 Statistical Significance
   - 4.9 Summary of Key Findings

5. [Discussion](sections/05_discussion.md)
   - 5.1 Interpretation of Results
   - 5.2 Novelty and Contributions
   - 5.3 Comparison with Related Work
   - 5.4 Limitations and Challenges
   - 5.5 Future Directions
   - 5.6 Broader Implications
   - 5.7 Ethical Considerations

6. [Conclusion](sections/06_conclusion.md)
   - 6.1 Summary of Work
   - 6.2 Key Contributions
   - 6.3 Impact and Significance
   - 6.4 Limitations and Future Work
   - 6.5 Broader Perspective
   - 6.6 Concluding Remarks

7. [References](sections/07_references.md)

8. [Appendices](sections/08_appendices.md)
   - Appendix A: Hyperparameter Details
   - Appendix B: Dataset Statistics
   - Appendix C: Confusion Matrices
   - Appendix D: Training Curves
   - Appendix E: Operating Threshold Analysis
   - Appendix F: Computational Requirements
   - Appendix G: Software Dependencies
   - Appendix H: Code Repository Structure
   - Appendix I: Reproducibility Checklist
   - Appendix J: Ethical Approval and Data Usage

---

## List of Figures

1. Figure 3.1: HAM10000 Dataset Class Distribution
2. Figure 3.2: Data Preprocessing Pipeline
3. Figure 3.3: Model Architecture Comparison
4. Figure 4.1: Training and Validation Learning Curves
5. Figure 4.2: Confusion Matrices for All Architectures
6. Figure 4.3: Reliability Diagrams Before and After Calibration
7. Figure 4.4: Grad-CAM Attention Maps for Melanoma Cases
8. Figure 4.5: Grad-CAM Attention Maps for Benign Cases
9. Figure 4.6: Inference Time Comparison
10. Figure 5.1: ROC Curves for Melanoma Detection

*(Figures are generated from training results and stored in thesis/figures/)*

---

## List of Tables

1. Table 3.1: Dataset Split Statistics
2. Table 3.2: Model Architecture Specifications
3. Table 4.1: Model Performance Summary
4. Table 4.2: Melanoma Detection Performance
5. Table 4.3: Temperature Calibration Results
6. Table 5.1: Comparison with Related Work

---

## Acknowledgments

This work was made possible by the availability of the HAM10000 dataset, released by Tschandl et al. through the International Skin Imaging Collaboration (ISIC). I thank the Medical University of Vienna and Cliff Rosendahl for their contributions to open dermatological data.

The research benefited from publicly available pretrained models and open-source software including PyTorch, torchvision, Gradio, and numerous Python libraries maintained by the scientific computing community.

---

## Declaration

I declare that this thesis represents my own work, conducted independently with appropriate acknowledgment of sources and assistance. The code implementation, experimental design, analysis, and writing are my original contributions. Previously published datasets and pretrained models are properly cited.

---

## Repository

Complete code, trained models, and documentation are available at:
**GitHub**: github.com/setab/user-facing-AI-powered-melanoma-detection-system-with-explainability

The repository includes:
- Full source code for training and inference
- Trained model checkpoints with calibration parameters
- Web interface implementation
- Comprehensive documentation
- Jupyter notebooks for analysis and visualization

---

## How to Read This Thesis

**For Quick Overview**: Read Abstract, Introduction (Section 1), and Conclusion (Section 6)

**For Technical Details**: Focus on Methodology (Section 3) and Results (Section 4)

**For Clinical Context**: Read Introduction (Section 1), Background (Section 2.1), and Discussion (Section 5.1)

**For Implementation**: See Methodology (Section 3), Appendices (Section 8), and code repository

**For Reproducibility**: Follow Appendix I checklist with code from repository

---

## Document Statistics

- **Total Pages**: ~80-100 pages (estimated with figures)
- **Word Count**: ~25,000 words
- **Sections**: 8 major sections
- **Figures**: 10+ visualization figures
- **Tables**: 15+ data tables
- **References**: 30 primary citations
- **Code Lines**: ~2,500 lines of Python
- **Experiments**: 4 architectures × 20 epochs = 80 training runs
- **Dataset**: 10,013 images across 7 classes

---

*This thesis documents a complete, reproducible research project from data processing through deployment, demonstrating that AI-assisted melanoma detection can achieve clinically-meaningful performance with calibrated probabilities and explainable predictions.*
