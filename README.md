# SketchXAI: Explainable AI for Human Sketches

SketchXAI is an explainable AI framework designed to introduce dual-engine transparency (spatial and temporal) into sketch classification. This repository contains the complete dataset pipeline, neural network architectures, and evaluation baselines used to validate the framework.

---

## 📁 Repository Directory Structure

Based on the project workspace layout, the repository is organized as follows:

```text
├── figures/                              # Comprehensive visual evaluations and system pipelines
│   ├── SLI_breakdown.png                 # Theoretical schema of Recovery & Transfer mechanics
│   ├── Streamlit_LIME.png                # UI implementation snapshot of spatial LIME masks
│   ├── Streamlit_LIME_2.png              # Secondary interface verification of LIME heatmaps
│   ├── Streamlit_SLI.png                 # UI snapshot highlighting temporal stroke-level ordering
│   ├── Streamlit_SLI_2.png               # Secondary interface verification of stroke order scores
│   ├── baseline_345_animal_slice.png     # 7-class animal contamination confusion matrix (345 classes)
│   ├── baseline_345_convergence.png      # Loss/accuracy curves for the 345-class baseline model
│   ├── class_convergence.png             # Training convergence trajectories for the 87-class model
│   ├── image_description.md              # Markdown asset mapping graphic metadata descriptions
│   ├── sketchxai_lime_step_by_step.png   # Step-by-step visual parsing of superpixel perturbation
│   ├── sketchxai_multiclass_metrics_ch...# Global performance metrics and classification charts
│   ├── sketchxai_preprocessing_pipeli... # Visual schema of scaling, reshaping, and normalization data pipeline
│   ├── sketchxai_tier1_global_silhouett...# Macro-level confusion matrix silhouette overview
│   └── sketchxai_tier2_animal_slice.png  # Curated zoom slice confusion matrix (87 classes)
├── notebooks/                            # Development and experimental environments
│   ├── 345_classes.ipynb                 # Scalability stress-test baseline codebase
│   └── SketchXAI_Training.ipynb          # Main 87-class framework training notebook
├── SKETCHXAI__EXPLAINABLE_AI_F...        # Manuscript documentation / project overview
├── app.py                                # Core user interface application script
├── labels.txt                            # Organized class vocabulary targets
└── sketchxai_matching_model.h5           # Compiled neural network weights file

```

---

## 🧠 Model Architecture Specification

To maintain a true controlled experiment across both the primary framework and the scalability stress-test baseline, both notebooks execute an identical convolutional neural network (CNN) design built with TensorFlow/Keras.

```text
Input Sketch (28x28x1)
       │
       ▼
 [Conv_Block_1]  ──► 30 Filters (3x3 kernel), ReLU Activation
       │
       ▼
  [MaxPool_1]    ──► 2x2 Spatial Downsampling
       │
       ▼
 [Conv_Block_2]  ──► 15 Filters (3x3 kernel), ReLU Activation
       │
       ▼
  [MaxPool_2]    ──► 2x2 Spatial Downsampling
       │
       ▼
   [Dropout]     ──► 20% Node Regularization (Overfitting Guard)
       │
       ▼
   [Flatten]     ──► 1D Vector Layer Transformation (375 Nodes)
       │
       ▼
[Dense_Hidden_1] ──► 128 Fully-Connected Neurons, ReLU Activation
       │
       ▼
[Dense_Hidden_2] ──► 50 Fully-Connected Neurons, ReLU Activation
       │
       ▼
[Softmax_Output] ──► Dense Layer mapped to active class capacity

```

### Parametric Layer Variations

* **Primary Framework (`SketchXAI_Training.ipynb`):** The final output layer is configured as `Dense(87, activation='softmax')`.
* **Scalability Baseline (`345_classes.ipynb`):** The final output layer is configured as `Dense(345, activation='softmax')`.

---

## 💾 GPU Memory Infrastructure (Handling Large-Scale Streaming)

Loading the QuickDraw bitmap array subsets straight into standard RAM causes immediate runtime crashes due to memory overflow. For instance, the 345-class baseline processes a massive pool of **1,725,000 images** ($345 \text{ classes} \times 5,000 \text{ samples}$).

SketchXAI completely bypasses host memory constraints using a two-tier hardware optimization pipeline:

1. **NumPy Memory-Mapping (`mmap_mode='r'`):** Instead of parsing whole `.npy` database blocks into system memory, the code opens disk-backed read-only file pointers. Arrays remain stored on disk, and specific data boundaries are only fetched during live training steps.
2. **On-the-Fly Generator Streams (`tf.data.Dataset.from_generator`):** Data processing is deferred to an active background streaming thread. The custom `shuffled_generator` pulls a tiny index block from the disk pointers, normalizes the pixel intensities to bounds of $[0.0, 1.0]$, shapes them into 2D spatial grids ($28 \times 28 \times 1$), and passes batches of exactly 128 images directly into GPU VRAM buffers. Memory usage remains completely flat throughout execution.

---

## ⏱️ Why Exactly 12 Epochs?

The maximum optimization lifecycle is strictly configured to **12 epochs** due to three main scientific constraints:

* **Experimental Standardization Control:** To scientifically prove that filtering the problem space down to a curated vocabulary was structurally superior, the baseline model and primary framework must be given the exact same training runway. This ensures that any accuracy changes are purely a product of dataset engineering rather than uneven optimization tracking.
* **Hardware Limit Constraints:** A single pass through the baseline dataset handles 1.7 million sketches per epoch. Capping the experiment at 12 epochs prevents Google Colab container runtimes from exceeding active GPU compute allocation bounds or triggering infrastructure timeouts before the `.h5` weights are saved locally.
* **Early Stopping Regularization Safety Net:** Both scripts embed an active `EarlyStopping` monitoring callback with a patience setting of 3 epochs. The 12-epoch ceiling serves as an appropriate limit because the internal loss curves reach an asymptotic plateau before this threshold is crossed.

---

## 📊 Discrepancy Clarification: 77.87% vs. ~73% Accuracy

During live local execution of the notebook `SketchXAI_Training.ipynb`, the training loop prints out a terminal convergence state of **$72.68\%$ validation accuracy** at Epoch 12, whereas the published SPECTRA 2026 paper documents a final convergence metric of **$77.87\%$**.

This is an expected variance caused by a deliberate runtime constraint:

* **The Cause:** The uploaded code cell uses a fixed ceiling of 12 epochs (`epochs=12`). Because validation accuracy was still actively climbing in the final frames (rising from $72.42\%$ in Epoch 10 to $72.63\%$ in Epoch 11, and landing at $72.68\%$ in Epoch 12), the network simply ran out of gradient descent updates.
* **The Resolution:** The higher metric of **$77.87\%$** published in the paper results from relaxing this maximum training ceiling to **30 epochs**. This allows the optimization loop to run continuously until it naturally activates the `EarlyStopping(patience=3)` callback on validation loss, which occurs at the **19th epoch** as detailed in the manuscript text. The underlying code logic and dataset parameters are otherwise mathematically identical.

---

## 🔍 The Dual-XAI Engine Explained

SketchXAI provides interpretability by approaching human sketches from two separate angles: spatial region importance and temporal stroke progression.

```text
                            ┌───────────────────┐
                            │    Human Sketch   │
                            └─────────┬─────────┘
                                      │
             ┌────────────────────────┴────────────────────────┐
             ▼                                                 ▼
   [Spatial Perturbation]                            [Temporal Progression]
             │                                                 │
             ▼                                                 ▼
 ┌───────────────────────┐                         ┌───────────────────────┐
 │    LIME Integration   │                         │ Stroke-Level Import.  │
 ├───────────────────────┤                         ├───────────────────────┤
 │ Superpixel segments   │                         │ Mask / shift ordered  │
 │ perturbed randomly to │                         │ stroke sequences to   │
 │ observe degradation   │                         │ quantify confidence   │
 │ in target prediction. │                         │ score deltas (ΔP).    │
 └───────────┬───────────┘                         └───────────┬───────────┘
             │                                                 │
             └────────────────────────┬────────────────────────┘
                                      ▼
                            ┌───────────────────┐
                            │ Unified Interface │
                            └───────────────────┘

```

### 1. Spatial Explainability: LIME Integration

LIME (Local Interpretable Model-agnostic Explanations) treats the trained CNN as a black-box classifier. It breaks the spatial layout of a sketch down into discrete superpixel segments. LIME then randomly turns these segments on or off, creating thousands of slightly modified variations of the original sketch, and passes them through the model to see how the prediction confidence shifts. By fitting a simple, interpretable linear model to these confidence changes, LIME generates high-contrast visual heatmaps. These heatmaps clearly highlight which specific pixels directly supported the target class prediction and which ones remained entirely neutral.

### 2. Temporal Explainability: Stroke-Level Importance (SLI)

While LIME analyzes static pixels, the core innovation of SketchXAI is Stroke-Level Importance (SLI), which evaluates sketches in their natural form as ordered sequences of drawn strokes. SLI determines importance by measuring changes in model confidence when strokes are systematically altered through two core features:

* **Recovery Mechanism:** This process isolated strokes by masking or modifying their locations on the canvas, measuring exactly how much the model's confidence dropped when that specific stroke vanished.
* **Transfer Mechanism:** This process dynamically shifted stroke features between different drawings to isolate and evaluate their direct structural influence.

The resulting importance score is calculated from the prediction probability delta:

$$I_{ij} \propto \frac{1}{\Delta P}$$

This mathematical relationship ensures that strokes causing the most significant drop in confidence when removed are correctly identified as the most critical features defining the human drawing.
