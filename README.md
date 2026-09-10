# PCOS Detection from Ultrasound Images (CNN)
Creating a convolutional neural network to predict PCOS from ultrasound images. Classifies ovarian ultrasound images as PCOS-positive or non-PCOS, trained and validated across two independent public datasets.

PCOS (Polycystic Ovarian Syndrome, now PMOS, or Polyendocrine Metabolic Ovarian Syndrome) affects an estimated 6–21% of reproductive-aged women, but diagnosis via ultrasound is subjective and heavily dependent on operator skill and equipment. This project explores whether an end-to-end CNN, trained directly on raw grayscale ultrasound images, can learn clinically meaningful, generalizable features of PCOS morphology — rather than dataset-specific artifacts.

### Key Result
The most important finding was what happened when the model was tested on data it had never seen:
Initial cross-dataset validation failed completely: a model trained on the Figshare dataset and tested on an independent Kaggle dataset collapsed to 40.5% accuracy with 0% recall. This showed the model had learned dataset-specific patterns, not real PCOS morphology. Adding batch normalization and switching to BCEWithLogitsLoss fixed this. The same architecture, retrained, jumped to 98.3% cross-dataset accuracy with 100% precision. This suggests the final architecture generalizes across imaging sources.

## Results
| Evaluation | Accuracy | Precision | Recall | ROC AUC |
| :--- | :---: | :---: | :---: | :---: |
| Figshare train | 94.3 | 92.0 | 94.9 | 0.987 |
| Figshare test | 97.3 | 95.7 | 97.8 | 0.995 |
| Cross dataset trained on Figshare tested on Kaggle before fix | 40.5 | 0 | 0 | 0.436 |
| Cross dataset trained on Figshare tested on Kaggle after fix | 98.3 | 100 | 97.1 | 0.999 |


## Datasets
- Kaggle — PCOS Detection Using Ultrasound Images (also used in Moral et al., 2024). 3,856 images, pre-split into train/test.
- Figshare — Indirani, 2024. 12,680 images compiled from multiple public sources; more diverse follicular presentation, used as the primary training set.

Both datasets were resized to 128×128 grayscale, augmented (horizontal/vertical flips, Gaussian blur), and re-split 70/30.

## Model Architecture
- Sequential CNN, 3 convolutional blocks + fully connected classification head:

| Block | Layers | Output size |
| :--- | :--- | :--- |
| Conv Block 1 | 2× Conv(32, 3×3) → BatchNorm → ReLU → Dropout(0.25) → MaxPool(2×2) | 128×128 → 62×62 |
| Conv Block 2 | Conv(32, 3×3) → BatchNorm → ReLU → Dropout(0.25) → MaxPool(2×2) | 62×62 → 30×30 |
| Conv Block 3 | Conv(12, 3×3) → BatchNorm → ReLU → Dropout(0.25) → MaxPool(2×2) | 30×30 → 14×14 |
| Head | Flatten (3,136) → FC(64) → ReLU → Dropout(0.5) → FC(1) | binary logit |

- Loss: BCEWithLogitsLoss (with pos_weight for class imbalance)
- Optimizer: Adam, lr = 0.0001, batch size = 64
- Epochs: 20

## Limitations
- Both datasets are public with limited documentation on acquisition methods; real clinical images may differ.
- Class imbalance was handled via pos_weight, which may not transfer to different positive/negative ratios in deployment.
- The model is a black box: it outputs a binary label with no indication of which image regions drove the prediction.

## Future Work
- Saliency mapping for interpretability
- Validation on real clinical (non-research) datasets
- Testing across more hospitals/imaging equipment
- Extending output beyond binary classification (e.g., follicle count estimation, borderline-case flagging)

## References
- Cross, J. L., Choma, M. A., & Onofrey, J. A. (2024). Bias in medical AI: Implications for clinical decision-making. PLOS Digital Health, 3(11), e0000651. https://doi.org/10.1371/journal.pdig.0000651
- Dewani, D., Karwade, P., & Mahajan, K. S. (2023). The Invisible Struggle: The Psychosocial aspects of Polycystic Ovary Syndrome. Cureus, 15(12), e51321. https://doi.org/10.7759/cureus.51321
- GeeksforGeeks. (2021). What is Saliency Map? GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/what-is-saliency-map/
- Kranthi, S., Sandeep, Y., Pranathi, K. . ., Lydia, E. L., Joshi, G. P., & Cho, W. (2026). A multimodal feature fusion with deep representation learning approach for polycystic ovary syndrome diagnosis using ultrasound images. Scientific Reports, 16(1). https://doi.org/10.1038/s41598-026-40718-w
- Mienye, I. D., Swart, T. G., Obaido, G., Jordan, M., & Ilono, P. (2025). Deep convolutional Neural Networks in Medical Image Analysis: A review. Information, 16(3), 195. https://doi.org/10.3390/info16030195
- Moral, P., Mustafi, D., Mustafi, A., & Sahana, S. K. (2024). CystNet: An AI driven model for PCOS detection using multilevel thresholding of ultrasound images. Scientific Reports, 14(1), 25012. https://doi.org/10.1038/s41598-024-75964-3
- PCOS detection using ultrasound images. (n.d.). https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images
- Polycystic ovary Syndrome (PCOS). (n.d.). Cleveland Clinic. https://my.clevelandclinic.org/health/diseases/8316-polycystic-ovary-syndrome-pcos
- Suha, S. A., & Islam, M. N. (2022). An extended machine learning technique for polycystic ovary syndrome detection using ovary ultrasound image. Scientific Reports, 12(1), 17123. https://doi.org/10.1038/s41598-022-21724-0
- Sundari, M. S., Sailaja, N. V., Swapna, D., Vikkurty, S., Jadala, V. C., Durga, K., & Thottempudi, P. (2025). Transfer learning-enhanced CNN model for integrative ultrasound and biomarker-based diagnosis of polycystic ovarian disease. Scientific Reports, 15(1), 34519. https://doi.org/10.1038/s41598-025-17711-w
- Sung, N., Amir, J., Alwahab, U. A., & Falcone, T. (2026). Polycystic ovary syndrome: An update on diagnosis and management. Cleveland Clinic Journal of Medicine, 93(3), 176–183. https://doi.org/10.3949/ccjm.93a.25090
- Zhang, L., Wang, X., Yang, D., Sanford, T., Harmon, S., Turkbey, B., Wood, B. J., Roth, H., Myronenko, A., Xu, D., & Xu, Z. (2020). Generalizing deep learning for medical image segmentation to unseen domains via deep stacked transformation. IEEE Transactions on Medical Imaging, 39(7), 2531–2540. https://doi.org/10.1109/tmi.2020.2973595
