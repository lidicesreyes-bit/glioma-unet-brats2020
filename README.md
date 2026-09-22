# Glioma Segmentation with 2D U-Net + BraTS2020

Code for our paper published at **CSNDSP 2026** (IEEE).

**Paper:** *Optimizing Magnetic Resonance Image Segmentation Through Scalable Deep Learning and Hierarchical Data Management*
**Venue:** 2026 15th International Symposium on Communication Systems, Networks and Digital Signal Processing (CSNDSP), Edinburgh, United Kingdom, pp. 1-6
**DOI:** [10.1109/CSNDSP68462.2026.11654508](https://doi.org/10.1109/CSNDSP68462.2026.11654508)

**Paper authors:** Lídices Reyes-Hung, Gabriel Trinke, Ismael Soto, J. Serey
**Code:** Lídices Reyes-Hung
**Institution:** University of Santiago, Chile (USACH)
**Contact:** lidices.reyes@usach.cl · gabriel.trinke@usach.cl · ismael.soto@usach.cl

---

## Overview

This repository contains the implementation of an efficient and scalable glioma segmentation system based on an optimized 2D U-Net architecture, integrated with an HDF5-based data engineering pipeline. The system enables training on large-scale multimodal MRI datasets without loading the entire dataset into memory, making it suitable for consumer-grade GPU hardware.

---

## Results

Reported on <!-- COMPLETAR: conjunto y partición usados para evaluar -->.

### Binary Segmentation (main model)

| Metric | Result |
|--------|--------|
| DSC | 0.884 |
| Sensitivity | 0.851 |
| Specificity | 0.992 |
| Hausdorff Distance | 4.2 mm |

### Multiclass Segmentation (per subregion)

| Subregion | DSC |
|-----------|-----|
| Peritumoral Edema | 0.878 |
| Necrotic Core | 0.825 |
| Enhancing Tumor | 0.912 |

---

## Repository Structure

```
glioma-unet-brats2020/
├── glioma.py                       # Main training script (Windows/Linux)
├── glioma_colab_final_v2.ipynb     # Google Colab notebook (binary model)
├── glioma_multiclass_colab.ipynb   # Google Colab notebook (multiclass model)
├── requirements.txt
└── README.md
```

---

## Requirements

```
Python 3.10+
TensorFlow 2.19
h5py
nibabel
scikit-learn
scipy
matplotlib
numpy
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Dataset

BraTS2020 — Brain Tumor Segmentation Challenge 2020
Available on Kaggle:
https://www.kaggle.com/datasets/awsaf49/brats2020-training-data

Each case contains 4 co-registered MRI modalities:
- T1
- T1ce (contrast-enhanced)
- T2
- FLAIR

---

## Training

### Option 1 — Google Colab (recommended)
1. Open `glioma_colab_final_v2.ipynb` in Google Colab
2. Set runtime to **T4 GPU**
3. Run cells in order

### Option 2 — Local (Windows/Linux)
```bash
pip install -r requirements.txt
python glioma.py
```

---

## Model Architecture

2D U-Net encoder-decoder with:
- Encoder: Conv blocks with filters 32 → 64 → 128 → 256
- Bottleneck: 512 filters
- Decoder: Transposed convolutions + skip connections
- Output: Sigmoid activation (binary) or per-class sigmoid (multiclass)
- Parameters: ~7.8 million
- Input: 128×128×4 (4 MRI modalities)

---

## Hardware Used

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA Tesla T4 (15 GB VRAM) |
| Platform | Google Colaboratory |
| Framework | TensorFlow 2.19 |
| Batch size | 16 |
| Epochs | 15 |
| Training slices | 57,195 |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{reyeshung2026glioma,
  author    = {Reyes-Hung, L. and Trinke, G. and Soto, I. and Serey, J.},
  title     = {Optimizing Magnetic
