<div align="center">

# 🧠 Brain Tumor Classification & Segmentation
### Using Machine Learning and Deep Learning on MRI Scans

*A comparative study of 6 classifiers and 4 segmentation architectures, with Grad-CAM explainability.*

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-orange)
![Task](https://img.shields.io/badge/Task-Classification%20%2B%20Segmentation-blueviolet)
![Dataset](https://img.shields.io/badge/Dataset-Figshare%20Brain%20Tumor-informational)
![Best Classifier](https://img.shields.io/badge/VGG16-97%25%20Accuracy-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

**B.Sc. in CSE Capstone Project · Bangladesh University of Business and Technology (BUBT) · October 2025**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Pipeline](#-pipeline)
- [Dataset](#-dataset)
- [Models](#-models)
- [Results in Detail](#-results-in-detail)
- [Explainability (Grad-CAM)](#-explainability-grad-cam)
- [Limitations & Future Work](#-limitations--future-work)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Team](#-team)
- [Contact](#-contact)
- [Citation](#-citation)
- [Disclaimer](#%EF%B8%8F-disclaimer)
- [License](#-license)

---

## 🔬 Overview

Brain tumors are among the most dangerous neurological conditions, and a misdiagnosis can lead to wrong treatment and lower survival chances. Manual MRI analysis is time-consuming, depends on radiologist experience, and is hard to scale given the shortage of specialists.

This project builds an **automated brain tumor analysis system** that does two things:

1. **Classifies** an MRI scan into one of three tumor types: **Glioma**, **Meningioma**, or **Pituitary**.
2. **Segments** the tumor region at the pixel level to show where it is.

We benchmark **three deep learning classifiers** (ResNet50, VGG16, DenseNet121) against **three classical ML classifiers** (SVM, Random Forest, KNN), and compare **four segmentation networks** (U-Net, FCN, DeepLabV3+, Attention U-Net). **Grad-CAM** is used to make the model's decisions interpretable.

### ✨ Highlights

- 🏆 **VGG16** is the best classifier, with **97% accuracy / F1** (98.2% in the detailed per-class test evaluation).
- 🎯 **U-Net** is the best segmentation model, with **98.59% pixel accuracy** and the best overlap scores (**IoU 12.80%, Dice 18.24%**).
- 📊 Deep learning beats classical ML by **17–19 percentage points** on the same task.
- 🔍 **Grad-CAM** heatmaps confirm the model focuses on tumor regions.
- 🧪 Regularization (augmentation, dropout, early stopping) keeps training and validation curves close, so there is no major overfitting.

---

## 🏁 Key Results

| Task | Best Model | Headline Metric |
|------|-----------|-----------------|
| **Tumor Classification** | **VGG16** (fine-tuned) | **97% accuracy, precision, recall, F1** |
| **Tumor Segmentation** | **U-Net** | **98.59% pixel accuracy, Dice 18.24%, IoU 12.80%** |
| **Explainability** | Grad-CAM on U-Net | Activation concentrated on tumor region |

---

## 🔄 Pipeline

<div align="center">
<img src="assets/methodology.png" alt="Overall methodology" width="620"/>
</div>

1. **Data collection**: public Figshare brain tumor dataset.
2. **Preprocessing**: resize to 256×256, min-max normalization to [0, 1].
3. **Augmentation**: random rotation (±15°), horizontal/vertical flips, zoom.
4. **Stratified split**: 75% train / 10% validation / 15% test.
5. **Classification**: ResNet50, VGG16, DenseNet121 + SVM, Random Forest, KNN.
6. **Feature engineering for classical ML**: Discrete Wavelet Transform (DWT) for feature extraction, Particle Swarm Optimization (PSO) for feature selection.
7. **Segmentation**: U-Net, FCN, DeepLabV3+, Attention U-Net.
8. **Explainability**: Grad-CAM heatmaps.

---

## 🗂 Dataset

| Property | Detail |
|----------|--------|
| **Source** | [Brain Tumor Dataset, Figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) |
| **Images** | 3,064 contrast-enhanced T1-weighted MRI slices |
| **Classes** | Glioma · Meningioma · Pituitary tumor |
| **Annotations** | Expert-annotated tumor masks |
| **Input size** | Resampled to 256 × 256 |
| **Split** | 75% train · 10% validation · 15% test (stratified) |

> The dataset is **not included** in this repository. Download it from the link above and place it as described in [Getting Started](#-getting-started).

---

## 🧩 Models

### Classification

| Family | Models |
|--------|--------|
| **Deep Learning (CNN)** | ResNet50 · VGG16 (fine-tuned, separable convolutions) · DenseNet121 |
| **Classical ML** | SVM · Random Forest · KNN (on DWT + PSO features) |

Loss: categorical cross-entropy · VGG16 trained for up to 50 epochs with early stopping on validation loss.

### Segmentation

| Model | Idea |
|-------|------|
| **U-Net** | Encoder-decoder with skip connections for precise localization |
| **FCN** | Fully convolutional, multi-scale upsampling (FCN-32s/16s/8s) |
| **DeepLabV3+** | Atrous convolution + encoder-decoder for multi-scale context |
| **Attention U-Net** | U-Net with attention gates to focus on tumor-relevant regions |

Loss: Binary Focal Cross-Entropy · Optimal probability threshold: **0.3**

---

## 📈 Results in Detail

### Classification Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|:--------:|:---------:|:------:|:--------:|
| **VGG16** 🥇 | **0.97** | **0.97** | **0.97** | **0.97** |
| ResNet50 | 0.91 | 0.91 | 0.91 | 0.91 |
| DenseNet121 | 0.89 | 0.89 | 0.89 | 0.89 |
| Random Forest | 0.80 | 0.80 | 0.80 | 0.80 |
| KNN | 0.78 | 0.81 | 0.78 | 0.77 |
| SVM | 0.78 | 0.78 | 0.78 | 0.78 |

**Per-class results for fine-tuned VGG16** (independent test set; overall accuracy 98.2%):

| Class | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| Glioma | 97.9% | 97.7% | 97.8% | 214 |
| Meningioma | 98.5% | 97.8% | 98.1% | 106 |
| Pituitary | 98.5% | 100% | 99.2% | 140 |

**Training curves**

<table>
<tr>
<td align="center"><b>VGG16</b><br><img src="assets/vgg16_curves.png" width="300"/></td>
<td align="center"><b>ResNet50</b><br><img src="assets/resnet50_curves.png" width="300"/></td>
<td align="center"><b>DenseNet121</b><br><img src="assets/densenet121_curves.png" width="300"/></td>
</tr>
</table>

**Takeaways**
- VGG16's training and validation curves converge closely, which indicates augmentation, dropout, and early stopping controlled overfitting.
- DenseNet121 likely overfits on this relatively small dataset despite its theoretical efficiency.
- Classical models (SVM, RF, KNN) trail deep models because handcrafted features capture less than learned hierarchical features.

### Segmentation Performance

| Model | Accuracy (%) | Loss | IoU (%) | Dice (%) |
|-------|:-----------:|:----:|:-------:|:--------:|
| **U-Net** 🥇 | **98.59** | 0.0109 | **12.80** | **18.24** |
| Attention U-Net | 98.01 | 0.0095 | 9.80 | 10.67 |
| FCN | 97.87 | **0.0085** | 10.55 | 14.87 |
| DeepLabV3+ | 95.70 | 0.0101 | 4.93 | 7.09 |

<div align="center">

**U-Net output: input → ground truth → raw probability map → binary mask (threshold = 0.3)**

<img src="assets/unet_segmentation_results.png" alt="U-Net segmentation results" width="800"/>

</div>

<table>
<tr>
<td align="center"><b>U-Net training</b><br><img src="assets/unet_training.png" width="320"/></td>
<td align="center"><b>Attention U-Net training</b><br><img src="assets/attention_unet_training.png" width="320"/></td>
</tr>
</table>

<details>
<summary><b>DeepLabV3+ comparison (click to expand)</b></summary>
<br>
<img src="assets/deeplabv3plus_results.png" alt="DeepLabV3+ results" width="800"/>
</details>

---

## 🔍 Explainability (Grad-CAM)

Grad-CAM highlights the regions that most influenced the model's decision. Activation concentrates around the tumor region, which supports that the model learns tumor-specific features and not background artifacts. This matters for clinical trust.

<div align="center">
<img src="assets/gradcam_visualization.png" alt="Grad-CAM visualization" width="800"/>
</div>

---

## ⚠️ Limitations & Future Work

We believe in reporting results honestly:

- **Overlap scores are low.** Pixel accuracy is high (~98%) mainly because tumors occupy a small fraction of each image (class imbalance). Dice and IoU are the more meaningful metrics, and they show the segmentation still needs improvement.
- **Single dataset.** Generalization across scanners, hospitals, and datasets is not yet validated.
- **2D slices only.** 3D volumetric context is not used.
- **Not clinically validated.** See the disclaimer below.

**Planned improvements**
- [ ] Improve Dice/IoU with Dice-focused or combined losses and better class-imbalance handling
- [ ] Multimodal MRI and 3D segmentation
- [ ] Cross-dataset validation (e.g., BraTS)
- [ ] Tumor volume and progression analysis
- [ ] A simple web/GUI tool for clinicians

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A GPU is recommended for training

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dataset setup

1. Download the dataset from [Figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427).
2. Place it in a `data/` folder in the project root.

### Run

```bash
# Open the notebooks in order
jupyter notebook
```

> Update the commands above to match your actual notebook/script names.

---

## 📁 Project Structure

```text
.
├── assets/                 # Figures used in this README
├── data/                   # Dataset (not tracked, download separately)
├── notebooks/              # Classification, segmentation, Grad-CAM notebooks
├── requirements.txt
├── LICENSE
└── README.md
```

> Adjust this tree to match your repository.

---

## 👥 Team

| Name |
|------|
| Raju Ahmad |
| Jarin Tasnim Lamia | 
| MD. Shahariar Hossain Khan | 
| Jannatul Ferdous Ananna | 
| Jahidul Islam | 


**Supervisor:** Sudipto Chaki, Assistant Professor, Department of CSE, BUBT

**Department of Computer Science and Engineering, Bangladesh University of Business and Technology (BUBT), Dhaka, Bangladesh**

---

## 📬 Contact

Have questions, feedback, or want to collaborate? Feel free to reach out:

**Raju Ahmad**
📧 [razuahmed1066@gmail.com](mailto:razuahmed1066@gmail.com)

You can also open an [issue](../../issues) on this repository for bugs or suggestions.

---

## 📚 Citation


If you use this work, please cite:

```bibtex
@thesis{ahmad2025braintumor,
  title   = {Human Brain Tumor Classification and Segmentation using Machine Learning and Deep Learning Algorithms},
  author  = {Ahmad, Raju and Lamia, Jarin Tasnim and Khan, MD. Shahariar Hossain and Ananna, Jannatul Ferdous and Islam, Jahidul},
  school  = {Bangladesh University of Business and Technology},
  address = {Dhaka, Bangladesh},
  year    = {2025},
  month   = oct,
  type    = {B.Sc. Thesis},
  note    = {Department of Computer Science and Engineering. Supervisor: Sudipto Chaki}
}
```

---

## ⚕️ Disclaimer

This project is for **research and educational purposes only**. It is **not a medical device** and must not be used for clinical diagnosis or treatment decisions. Always consult qualified medical professionals.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

---

<div align="center">

⭐ If you found this project useful, please consider giving it a star!

</div>
