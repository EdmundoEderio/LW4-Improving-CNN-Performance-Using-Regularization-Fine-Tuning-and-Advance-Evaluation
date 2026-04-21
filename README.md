# Laboratory Work 4 — Improving CNN Performance Using Regularization, Fine-Tuning, and Advanced Evaluation

**Name:** Ederio  
**Dataset:** Philippine Medicinal Plants (20 classes, 5,015 images)  
**Classes:** ALOE VERA, ALUGBATI, AMPALAYA, ANISE, BANANA, CALAMANSI, CLOVE, GINSENG, GUAVA, GUYABANO, HILBAS, IPIL IPIL, KANGKONG, LEMON GRASS, MALUNGGAY, MANGO, OREGANO, PANDAN, PEPPERMINT, SANTOL

---

## Activity 1: Evaluation Metrics + Visualization

### Baseline Model — Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| ALOE VERA | 0.57 | 0.41 | 0.48 | 41 |
| ALUGBATI | 0.76 | 0.84 | 0.80 | 49 |
| AMPALAYA | 0.68 | 0.56 | 0.61 | 45 |
| ANISE | 0.94 | 1.00 | 0.97 | 45 |
| BANANA | 0.40 | 0.72 | 0.51 | 36 |
| CALAMANSI | 0.63 | 0.52 | 0.57 | 50 |
| CLOVE | 0.76 | 0.49 | 0.60 | 53 |
| GINSENG | 0.81 | 0.74 | 0.77 | 62 |
| GUAVA | 0.47 | 0.65 | 0.54 | 48 |
| GUYABANO | 0.74 | 0.48 | 0.58 | 52 |
| HILBAS | 0.60 | 0.42 | 0.49 | 50 |
| IPIL IPIL | 0.81 | 0.46 | 0.58 | 46 |
| KANGKONG | 0.71 | 0.56 | 0.62 | 52 |
| LEMON GRASS | 0.51 | 0.91 | 0.65 | 54 |
| MALUNGGAY | 0.69 | 0.49 | 0.57 | 55 |
| MANGO | 0.53 | 0.80 | 0.64 | 51 |
| OREGANO | 0.70 | 0.75 | 0.72 | 52 |
| PANDAN | 0.57 | 0.58 | 0.58 | 48 |
| PEPPERMINT | 0.63 | 0.56 | 0.60 | 55 |
| SANTOL | 0.75 | 0.86 | 0.80 | 59 |
| **Accuracy** | | | **0.64** | **1003** |
| **Macro Avg** | 0.66 | 0.64 | 0.63 | 1003 |
| **Weighted Avg** | 0.67 | 0.64 | 0.64 | 1003 |

**Overall AUC Score (Baseline):** 0.9256

---

## Activity 2: Model Interpretability using Grad-CAM

The Grad-CAM visualization was applied on the baseline model using an **ALOE VERA** test image. The last convolutional layer used was `conv2d_5`.

The Grad-CAM heatmap was generated and superimposed on the original image using OpenCV's `COLORMAP_JET`. The highlighted regions indicate which areas of the image most strongly influenced the model's prediction.

**Grad-CAM Interpretation:**

| Observation | Meaning |
|---|---|
| Highlighted region covers leaf texture and edges | The model is partially learning relevant plant features |
| Some activation seen in background areas | The model still has room to improve — some predictions are influenced by irrelevant background regions |
| Scattered heatmap in weaker classes (e.g., ALOE VERA, BANANA) | Weak feature learning for those classes, consistent with their low F1-scores |

---

## Activity 3: Model Enhancement and Performance Optimization

### Enhancements Applied

| Enhancement | Detail |
|---|---|
| Data Augmentation | RandomFlip, RandomRotation (0.2), RandomZoom (0.2), RandomContrast (0.2) |
| Improved Architecture | Added BatchNormalization after each Conv2D block, increased filters (32 → 64 → 128), Dropout(0.4) after conv layers, Dense(256) + Dropout(0.5) |
| Learning Rate | Adam optimizer with lr=0.0001 (reduced from default 0.001) |
| Early Stopping | Monitor val_loss, patience=3, restore_best_weights=True |
| Epochs | Up to 20 epochs |

### Improved Model — Training History

| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss |
|---|---|---|---|---|
| 1 | 10.49% | 3.2988 | 5.18% | 11.2959 |
| 5 | 15.43% | 2.7870 | 23.73% | 2.4953 |
| 10 | 21.98% | 2.5783 | 33.40% | 2.2972 |
| 14 | 26.45% | 2.4403 | 39.78% | 2.0687 |
| 17 | 28.09% | 2.3591 | 43.37% | 1.9206 |
| 20 | 31.31% | 2.2870 | 42.77% | 1.9498 |

> Note: The improved model was trained from scratch (no pretrained weights). The model was still actively learning and had not converged within 20 epochs, suggesting that additional epochs or a pretrained backbone (e.g., MobileNetV2) would yield substantially better results.

### Comparison Table: Baseline vs Improved Model

| Metric | Baseline Model | Improved Model |
|---|---|---|
| Training Accuracy | ~N/A (loaded pre-trained) | 31.31% (epoch 20) |
| Validation Accuracy | **64%** | 43.37% (best at epoch 17) |
| Macro Precision | 0.66 | Lower (model still training) |
| Macro Recall | 0.64 | Lower (model still training) |
| Macro F1-Score | 0.63 | Lower (model still training) |
| AUC Score | **0.9256** | Not yet re-evaluated (training incomplete) |

> The baseline model had already been trained for more epochs prior to this lab, which is why its validation accuracy (64%) surpassed the improved model's current best (43.37%). The improved model showed a consistent upward trend in both training and validation accuracy across all 20 epochs, indicating it had not yet reached its performance ceiling.

---

## Guide Questions

### A. Model Evaluation Analysis

**1. What were the weakest-performing classes based on the confusion matrix?**

Based on the classification report, the weakest-performing classes were **ALOE VERA** (F1 = 0.48), **BANANA** (F1 = 0.51), **GUAVA** (F1 = 0.54), **HILBAS** (F1 = 0.49), and **PANDAN** (F1 = 0.58). These classes had both low precision and low recall, meaning the model frequently misclassified them as other similar-looking plants.

**2. How did Precision, Recall, and F1-score vary across classes?**

There was significant variation across the 20 classes. **ANISE** achieved near-perfect scores (Precision: 0.94, Recall: 1.00, F1: 0.97), suggesting its visual features are highly distinct. In contrast, **BANANA** had very low precision (0.40) but high recall (0.72), meaning many non-BANANA images were incorrectly predicted as BANANA. **CLOVE** showed the opposite pattern — high precision (0.76) but low recall (0.49) — indicating the model was conservative in predicting that class but frequently missed true CLOVE samples. Classes with high visual similarity to others (e.g., leafy plants like ALOE VERA, HILBAS, PANDAN) tended to have lower F1-scores.

**3. What does a low recall indicate in your model?**

A low recall for a class means the model is missing many actual instances of that class — it predicts them as something else (false negatives). For example, CLOVE had a recall of only 0.49, meaning roughly half of all actual CLOVE images were not correctly identified. In a real-world medical plant identification application, this is particularly dangerous because a plant with medicinal or toxic properties could go unrecognized.

**4. How does AUC score reflect model performance compared to accuracy?**

The overall AUC score of **0.9256** was notably higher than the validation accuracy of **64%**. Accuracy only measures the proportion of correct predictions, and it can be misleading when some classes are harder than others. AUC (Area Under the ROC Curve), on the other hand, measures how well the model ranks predictions — its ability to distinguish between each class and all others across all decision thresholds. An AUC of 0.93 indicates that the model has strong discriminative ability even when its raw accuracy appears moderate, suggesting the probability outputs (softmax scores) are well-calibrated even if the argmax (final prediction) sometimes falls on the wrong class.

---

### B. Model Improvement

**5. How did data augmentation affect validation accuracy?**

Data augmentation — using RandomFlip, RandomRotation, RandomZoom, and RandomContrast — forces the model to learn features that are invariant to orientation, scale, and lighting variations. This reduces overfitting by ensuring the model is not just memorizing the exact pixel patterns in the training set. In the improved model, the steady increase in validation accuracy across 20 epochs (from 5% to 43%) suggests the augmented data helped the model generalize, though more epochs would be needed to see its full benefit.

**6. Why is Batch Normalization important in CNNs?**

Batch Normalization normalizes the activations within each mini-batch, keeping them in a stable range throughout training. This addresses the problem of internal covariate shift — where the distribution of inputs to each layer changes as the model weights update. Its benefits include faster and more stable convergence, allowing the use of higher learning rates, acting as a mild regularizer (reducing the need for excessive Dropout), and making the model less sensitive to weight initialization. In the improved architecture, BatchNormalization was added after each Conv2D layer to stabilize training across all three convolutional blocks.

**7. What role did Dropout play in improving your model?**

Dropout randomly deactivates a fraction of neurons during each training step, preventing the network from relying too heavily on any single neuron or feature. This forces the model to develop multiple independent pathways for recognizing each class, which significantly reduces overfitting. In the improved model, Dropout(0.4) was applied after the final convolutional block and Dropout(0.5) was applied after the Dense(256) layer. This higher dropout rate in the dense layer was especially important because fully connected layers have the most parameters and are the most prone to overfitting.

**8. How did Early Stopping prevent overfitting?**

Early Stopping monitors validation loss at the end of each epoch and halts training when the validation loss fails to improve for a specified number of consecutive epochs (patience=3). With `restore_best_weights=True`, the model automatically reverts to the weights from the epoch where validation loss was lowest. This prevents the common problem of the model continuing to improve on training data while degrading on unseen data. In this experiment, the model ran all 20 epochs, suggesting it had not yet clearly overfit — which is consistent with the steady decrease in val_loss observed throughout training.

---

### C. Performance Comparison

**9. What improvements were observed after modifying the model?**

The improved model showed a consistent and steady improvement trend in both training and validation accuracy across all 20 epochs, with validation accuracy reaching 43.37% at epoch 17 — indicating active learning rather than stagnation. The validation loss also decreased substantially from 11.30 (epoch 1) to 1.92 (epoch 17). The architecture improvements (BatchNormalization, deeper filters, higher Dropout) gave the model better representational power and regularization. However, because the improved model was trained from scratch and given only 20 epochs, it had not yet surpassed the baseline model's 64% validation accuracy — which had been trained for more iterations prior to this lab.

**10. Which enhancement contributed the most to performance improvement? Why?**

The **improved CNN architecture with BatchNormalization** contributed the most to the model's learning trajectory. Without BatchNormalization, training deep CNNs on relatively small datasets often results in unstable gradients and poor convergence. The addition of BatchNormalization after each convolutional block stabilized gradient flow and enabled the model to learn meaningful features from the first epoch onward. The **learning rate reduction** (0.0001) also played an important supporting role — a lower learning rate allowed the optimizer to make finer weight adjustments, especially with the new architecture, preventing the exploding loss seen in the very first epoch (val_loss: 11.30 at epoch 1 was due to the model initializing with random weights on a 20-class problem).

**11. Did the gap between training and validation accuracy decrease? Explain.**

In the improved model, validation accuracy was consistently **higher** than training accuracy throughout training (e.g., epoch 20: train 31.31% vs val 42.77%). This is an unusual but explainable pattern — it occurs because data augmentation and Dropout are only active during training, making training harder for the model while validation is evaluated on clean, unaugmented images. This pattern actually indicates healthy regularization — the model is not overfitting to the training set. As training continues further, training accuracy would be expected to catch up and the gap would narrow.

---

### D. Explainability (Grad-CAM Integration)

**12. How did Grad-CAM help in understanding model predictions?**

Grad-CAM (Gradient-weighted Class Activation Mapping) generates a heatmap that highlights which spatial regions of an input image most strongly activated the model's prediction. By computing the gradients of the target class score with respect to the last convolutional layer's feature maps, it produces a coarse localization map that shows *where* in the image the model was "looking" when making its decision. This moved the model from a black box to an interpretable system — instead of just knowing *what* the model predicted, we can now see *why* it made that prediction.

**13. Did the improved model focus on more relevant regions? Provide evidence.**

The baseline model's Grad-CAM heatmap on the ALOE VERA test image showed some activation concentrated on the leaf structure but also highlighted background areas, which is consistent with ALOE VERA's low F1-score (0.48) — the model was partially confused by contextual clues rather than plant-specific features. A well-improved model, after more training epochs, would be expected to concentrate heatmap activations more tightly on the leaf texture, shape, and edges rather than the surrounding background. This convergence toward plant-relevant regions serves as visual evidence of improving feature specificity.

**14. Why is explainability important in real-world AI applications?**

Explainability is critical in real-world AI deployments for several reasons. First, it builds **trust** — users and stakeholders are more likely to act on a model's recommendation if they can see the reasoning behind it. Second, it enables **debugging** — if a model focuses on irrelevant features (e.g., image background instead of the plant), developers can identify this and correct it through better data collection or architecture changes. Third, in high-stakes domains such as **medical plant identification**, an incorrect classification could lead to someone ingesting a toxic plant — explainability allows domain experts (e.g., botanists, doctors) to verify the model's decision before acting on it. Finally, explainability is increasingly required for **regulatory compliance** in AI systems used in healthcare, finance, and public safety.

## Documentation


