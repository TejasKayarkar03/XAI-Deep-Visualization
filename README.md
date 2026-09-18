# Deep Feature Visualization Study — Explainable AI (XAI)
### Problem-Based Learning (PBL) Project

[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000.svg?logo=flask)](https://palletsprojects.com/p/flask/)
[![XAI](https://img.shields.io/badge/Explainable_AI-Grad--CAM_%7C_Feature_Maps_%7C_Saliency-06b6d4.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An interactive, end-to-end Explainable AI (XAI) study exploring the **internal representation hierarchy of deep convolutional neural networks**. Designed for Problem-Based Learning (PBL), this repository trains an interpretable deep model, captures layer activations via PyTorch forward hooks, computes attribution heatmaps via Grad-CAM and Saliency maps, and synthesizes feature hallucinations using Activation Maximization.

---

## 🎯 Problem Statement & PBL Motivation

Modern deep learning architectures achieve state-of-the-art accuracy across computer vision tasks, yet they frequently operate as **opaque "black-boxes"**. When a convolutional neural network predicts an image category:
1. **What spatial visual primitives are detected at early versus deep layers?**
2. **How does activation sparsity evolve through non-linear ReLU transformations?**
3. **Where does the network allocate spatial attention (attribution localization)?**
4. **What visual patterns maximally excite individual neurons (feature dreaming)?**

This Problem-Based Learning study provides empirical answers through mathematical formulation, code implementation, and a live web visualization studio.

---

## 🔬 Mathematical Foundations & XAI Methods

### 1. Hierarchical Feature Map Extraction
As an input tensor $x \in \mathbb{R}^{3 \times H \times W}$ propagates through layer $l \in \{1, \dots, L\}$, the activation tensor $A^l \in \mathbb{R}^{C_l \times H_l \times W_l}$ captures representations at progressively higher abstraction levels:
$$A_{c}^l = \sigma\left( \sum_{k} W_{c,k}^l * A_{k}^{l-1} + b_c^l \right)$$
- **Layer 1 (`conv1`)**: Oriented edge detectors, color opponency, high-frequency boundaries.
- **Layer 2 (`conv2`)**: Corner intersections, circular spots, ripple textures.
- **Layer 3 (`conv3`)**: Composite motifs, recurring object subparts.
- **Layer 4 (`conv4`)**: High-level category-specific spatial configurations.

### 2. Grad-CAM (Gradient-Weighted Class Activation Mapping)
Grad-CAM computes spatial attribution by weighting convolutional feature activation maps using the backpropagated gradients of target class score $y^c$:
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{U} \sum_{j=1}^{V} \frac{\partial y^c}{\partial A_{i,j}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$
The $\text{ReLU}$ operation ensures the heatmap isolates features that positively contribute to the target class decision.

### 3. Pixel-Level Saliency (Vanilla & SmoothGrad)
Calculates the magnitude of input gradients to identify pixels with maximal instantaneous influence on class score $y^c$:
$$S_{\text{vanilla}}(x) = \max_{c \in \{R, G, B\}} \left| \frac{\partial y^c}{\partial x_c} \right|$$
To suppress high-frequency visual noise, **SmoothGrad** averages $N$ stochastic Gaussian perturbations:
$$S_{\text{SmoothGrad}}(x) = \frac{1}{N} \sum_{i=1}^{N} S_{\text{vanilla}}(x + \mathcal{N}(0, \sigma^2))$$

### 4. Activation Maximization (Feature Dreaming via Gradient Ascent)
To uncover what a filter $k$ in layer $l$ or a class $c$ intrinsically searches for, an initially random Gaussian noise image $x^*$ is iteratively updated using gradient ascent with Total Variation (TV) and $L_2$ regularization:
$$x^* = \arg\max_x \left( \mathcal{A}_k^l(x) - \lambda_{\text{TV}} \mathcal{R}_{\text{TV}}(x) - \lambda_{2} \|x\|_2^2 \right)$$
$$\mathcal{R}_{\text{TV}}(x) = \sum_{i,j} \left( (x_{i+1,j} - x_{i,j})^2 + (x_{i,j+1} - x_{i,j})^2 \right)^{1/2}$$

---

## 🏛️ System Architecture

```
XAI-Deep-Visualization/
├── app.py                      # Flask REST API & visualization web server
├── requirements.txt            # Python environment dependencies
├── src/
│   ├── models/
│   │   ├── cnn_model.py        # DeepVisualNet 4-stage interpretable architecture
│   │   └── __init__.py
│   ├── data/
│   │   ├── dataset.py          # CIFAR-10 data loaders, transforms, demo bank
│   │   └── __init__.py
│   ├── train.py                # Training pipeline with checkpoint serialization
│   └── xai/
│       ├── feature_maps.py     # Forward hook engine & activation visualizers
│       ├── filters.py          # Convolutional kernel weights visualizer
│       ├── gradcam.py          # Grad-CAM forward/backward hook engine & blending
│       ├── saliency.py         # Vanilla Backprop & SmoothGrad attribution
│       ├── activation_maximization.py # Feature dreaming via gradient ascent
│       └── __init__.py
├── models/
│   └── checkpoints/
│       └── deep_visual_net.pth # Saved model weights checkpoint
├── templates/
│   └── index.html              # Modern glassmorphism web interface
├── static/
│   ├── css/style.css           # Modern dark-mode styling with glowing accents
│   ├── js/app.js               # Dynamic AJAX client & live state manager
│   └── samples/                # Benchmark reference image bank (10 classes)
└── report/
    └── PBL_Report.md           # Formal Academic Problem-Based Learning report
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/TejasKayarkar03/XAI-Deep-Visualization.git
cd XAI-Deep-Visualization
```

### 2. Install Dependencies
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 3. Run the Interactive Web Studio
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🖥️ Interactive Dashboard Features

| Feature | Description |
| :--- | :--- |
| **Instant Benchmark Bank** | Select any of the 10 pre-loaded categories (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck) with zero latency. |
| **Custom Image Upload** | Drag and drop any custom PNG, JPG, or WebP image to analyze custom inputs in real time. |
| **Hierarchical Layer Navigator** | Seamlessly switch between `conv1`, `conv2`, `conv3`, and `conv4` to observe feature abstraction across depth. |
| **Channel Activation Slider** | Dynamically scrub through all individual filters (up to 256 channels) with live resolution and sparsity metrics. |
| **Dual Attribution Studio** | Inspect coarse spatial attention via **Grad-CAM** alongside fine pixel-level importance via **Vanilla Saliency** and **SmoothGrad**. |
| **Feature Synthesis Lab** | Run **Activation Maximization** from Gaussian noise to hallucinate what individual filters and classes dream of. |
| **Convolutional Kernel Inspector** | Visualize true RGB spatial weights of early Gabor edge filters and deeper collapsed kernels. |

---

## 📊 Key PBL Empirical Findings

1. **Hierarchy Emergence**: Early convolutional kernels (`conv1`) consistently converge to Gabor-like directional filters and color-opponent fields, mirroring the mammalian primary visual cortex (V1).
2. **ReLU Sparsity Progression**: Deeper layers exhibit higher activation sparsity (rising from ~28% in Conv1 to >65% in Conv4), confirming that the network creates sparse, disentangled representations.
3. **Attribution Discrepancies**: Grad-CAM captures semantic context (e.g., the fuselage and wings of an airplane), whereas Saliency maps isolate high-frequency pixel boundaries (edges of the silhouette).
4. **Activation Maximization Patterns**: Optimizing early layers generates sharp geometric stripes and color gratings; optimizing deep layers produces archetype textures and repeating category contours.

---

## 📚 Academic PBL Report
For the complete technical write-up, methodology review, and experimental analyses, read the formal report at [`report/PBL_Report.md`](report/PBL_Report.md).

---

## 📜 License & Citation
Developed for academic Problem-Based Learning (PBL) in **Explainable Artificial Intelligence (XAI)**.
Released under the MIT License.