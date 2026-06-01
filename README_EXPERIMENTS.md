# Osteosarcoma Histopathology Experimental Pipeline

This directory contains the advanced deep learning pipeline used to conduct research on Osteosarcoma tumor classification. The experimental scope evaluates the performance of `ResNet-18` and `ResNet-50` under varying data distributions (Original vs. Balanced via Augmentation) and variable epochs.

The system is highly modularized into three core execution stages. 

---

## Technical Deep-Dive

### S01: Offline Dataset Balancing & Targeted Augmentation (`S01_GenerateBalancedDataset.m`)

**Purpose:** 
Medical datasets inherently suffer from severe class imbalances due to the biological prevalence of viable over non-viable/non-tumorous tissues. Left unchecked, Convolutional Neural Networks (CNNs) exhibit biased priors towards the majority class. This script algorithmically neutralizes the imbalance without utilizing duplicate sampling or under-sampling.

**Mechanisms & Logic:**
1. **Target Identification:** Analyzes the `dataset_original/train` split and calculates the deficit required for each minority class to exactly match the target of `1078` samples.
2. **Deterministic Folder Forking:** Copies the base architecture into `dataset_balanced` to ensure the original samples remain pristine and statistically reproducible. It also creates a strict standalone `dataset_augmentation` fork which *only* holds the newly synthesized artificial tissues.
3. **Advanced Mathematical Transformations:** Uses native MATLAB imaging permutations inside the locally declared `applyCustomAugmentation()` function:
   - **Affine Spatial Transforms:** Employs Bi-linear cropping rotations (`[-30, -20, -10, 10, 20, 30]`), X/Y-axis translations (`±5 pixels`), and scaling (`0.9x – 1.1x`). 
   - **Padding/Resolution Restoration:** Automatically corrects scaling boundary mismatch via uniform 0-tensor zero-padding, ensuring subsequent CNN input size match (`[224, 224, 3]`).
   - **Photometric Jitter:** Maps the RGB matrix into an HSV (Hue, Saturation, Value) geometric space. Manipulating the `V/Value` and `S/Saturation` matrices applies synthetic chemical slide staining diversity (brightness/contrast mutations), effectively neutralizing variances originating from non-uniform microscopic lighting during real-world hospital acquisition.
4. **Data Iteration:** Stores synthesized tensors using standard sequence padding (`aug_0001.png`), providing an absolute trace of the generated variants.

---

### S02: Deep Transfer Learning & Metric Generation (`S02_RunExperiments.m`)

**Purpose:** 
Executes the analytical blueprint (`PLAN.md`) sequentially spanning four experiment nodes (`E1` to `E4`). This script evaluates architectural depth (`ResNet-18` vs `ResNet-50`) combined with the effects of offline pre-balanced datasets against real-world imbalanced baselines over multiple time domains (1 to 20 epochs).

**Mechanisms & Logic:**
1. **Dynamic Architecture Hooking:** Instead of building CNNs from scratch, the script uses weights pre-trained on ImageNet. It computationally mutates the Directed Acyclic Graph (DAG via `layerGraph`):
   - Locates and disconnects the terminal 1000-class associative linear layer (`fc1000`).
   - Re-wires it to a specifically configured `fullyConnectedLayer` of `numClasses = 3`. 
   - Implements aggressive customized Learning Rate coefficients (`WeightLearnRateFactor=10`, `BiasLearnRateFactor=10`) causing the newly appended layer to heavily adapt over network backpropagation, while freezing earlier convolutional features.
2. **Asynchronous Batched Data Pipeline:** Wraps datasets within `augmentedImageDatastore` merely to handle uniform dimension resizing `[224, 224]`, allowing batch iterations to natively flow into the GPU.
3. **Training Execution Strategy:** Iterates loops defined by an `epochsList`. Uses the biologically adept **ADAM (Adaptive Moment Estimation)** momentum-optimizing stochastic gradient descent. 
4. **Statistical Metric Aggregation:** Eliminates dependency on basic *Accuracy* (which is skewed by imbalance).
   - Dynamically renders the `YTest` ground truth against `YPred` to establish a `confusionmat`.
   - Iterates row/col algebra extracting exact `TP`, `FP`, and `FN`.
   - Computes denominator-protected **Precision**, **Recall**, and the harmonic mean (**F1_Score**). 
5. **CSV Serialization & Model Persistence:** The resultant metrics per-epoch run are non-destructively streamed (append-mode) directly into `experiments_results.csv`, establishing data-frames suitable for scientific publication via Python/R. Each transient model state is snapshotted to a `.mat` binary (`model_E1...ep20.mat`) preventing computation loss.

---

### S03: Visual Interpretability & Traceability (`S03_ExplainabilityGradCAM.m`)

**Purpose:** 
Mitigates the "black-box" dilemma of Deep Learning in medical domains using **Gradient-weighted Class Activation Mapping (Grad-CAM)**. Instead of blindly trusting the algorithm, this script interrogates the model's internal convolutional weights to verify what specific pathological shapes (e.g., cell necrotization, hyperchromatic nuclei) triggered the final mathematical classification decision.

**Mechanisms & Logic:**
1. **Interactive Integration:** Utilizes `uigetfile` graphical hooks so users can manually inject any saved `.mat` snapshot from S02 along with random pathology `Test` images.
2. **Topological Feature Sub-routing:** Employs an arrayfun map function to automatically trace and identify the terminal Rectified Linear Unit (`ReLU`) mapping paths.
   - Targeting `res5c_relu` (in ResNet-50) or `res5b_relu` (in ResNet-18) allows the script to gather the raw high-level semantic tensors before spatial pooling destroys positional location data.
3. **Gradient Computation Activation:** Computes the derivative of the raw final class score with respect to the `FeatureLayer` maps. The resultant positive influences act as weights to generate a coarse localization heatmap.
4. **Alpha Matrix Blending (Superposition):** Employs standard spatial visualization (`imagesc`) scaling matrices to project the heatmap gradient into a pseudo-color spectrum (`jet`), and establishes transparency (`AlphaData=0.5`).
5. **Output Canvas:** Visualizes the Original Slide and the Overlaid Grad-CAM side-by-side, providing doctors or reviewers explicitly explainable visual proof of network logic.