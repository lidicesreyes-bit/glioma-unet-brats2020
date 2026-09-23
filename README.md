# Glioma Segmentation with 2D U-Net + BraTS2020

Source code for our paper published at **CSNDSP 2026** (IEEE). This is the repository referenced in the paper.

**Paper:** *Optimizing Magnetic Resonance Image Segmentation Through Scalable Deep Learning and Hierarchical Data Management*
**Venue:** 2026 15th International Symposium on Communication Systems, Networks and Digital Signal Processing (CSNDSP), Edinburgh, United Kingdom, pp. 1-6
**DOI:** [10.1109/CSNDSP68462.2026.11654508](https://doi.org/10.1109/CSNDSP68462.2026.11654508)

**Paper authors:** Lídices Reyes-Hung, Gabriel Trinke, Ismael Soto, Joel Serey
**Code:** Lídices Reyes-Hung
**Institution:** CIMTT, Department of Electrical Engineering, University of Santiago, Chile (USACH)
**Contact:** lidices.reyes@usach.cl · gabriel.trinke@usach.cl · ismael.soto@usach.cl · joel.serey@usach.cl

---

## Overview

An efficient and scalable glioma segmentation system based on an optimized 2D U-Net, combined with an HDF5 data-engineering layer using chunking and lazy loading. The dataset exceeds 50 GB, so the generator loads only the mini-batches needed at each step instead of the whole dataset. This cuts RAM usage by about 85% and keeps peak GPU memory under 8 GB. The model has ~7.8M trainable parameters, against 30M+ for a typical 3D U-Net.

---

## Repository structure

```
glioma-unet-brats2020/
├── glioma_colab_final_v2.ipynb     # Colab notebook, binary model — produces the published metrics
├── glioma_multiclass_colab.ipynb   # Colab notebook, multiclass model (3 subregions)
├── glioma.py                       # Local adaptation of the same pipeline (Windows/Linux)
├── requirements.txt
└── README.md
```

---

## Dataset

BraTS2020 — Brain Tumor Segmentation Challenge 2020, pre-processed HDF5 version:
https://www.kaggle.com/datasets/awsaf49/brats2020-training-data

Four co-registered modalities per case (T1, T1ce, T2, FLAIR). Volumes are 240×240×155 and are resized to 128×128×4 slices. Each `.h5` file holds one slice.

The Colab notebooks download the dataset directly from Kaggle, so you need your own `kaggle.json` API token. For local runs, download the dataset manually and set `DATA_DIR` in `glioma.py`.

---

## Data split

57,195 pre-processed HDF5 slices: **42,895 training / 8,580 validation / 5,720 test**.

Split: 10% test, then 15/90 of the remainder for validation (75/15/10), applied with `train_test_split` (seed 42) over the individual `.h5` slice files, **not grouped by patient**. Slices from the same patient may therefore appear in more than one split. The same split is used in the notebooks and in `glioma.py`.

---

## Results

Measured on the 5,720 held-out test slices.

### Binary segmentation (main model)

| Metric | Result |
|--------|--------|
| DSC | 0.884 |
| Sensitivity | 0.851 |
| Specificity | 0.992 |
| Hausdorff distance | 4.2 mm |

### Multiclass segmentation (per subregion)

| Subregion | DSC |
|-----------|-----|
| Peritumoral edema | 0.878 |
| Necrotic core | 0.825 |
| Enhancing tumor | 0.912 |

### Comparison with related methods

| Method | Dataset | DSC | Params |
|--------|---------|-----|--------|
| Wan et al. | LGG | 0.944 | ~40M |
| Montaha et al. | BraTS2020 | 0.930 | N/R |
| Bianconi et al. | BraTS2021 | 0.911 | ~25M |
| **This work** | **BraTS2020** | **0.884** | **7.8M** |
| Zhong et al. | BraTS2020 | 0.850 | 5.23M |

Class imbalance in the training set: background 98.24%, peritumoral edema 0.98%, necrotic core 0.52%, enhancing tumor 0.26%. A Dice loss is used instead of cross-entropy, which would otherwise collapse toward the background class.

---

## Requirements

```
Python 3.10+
TensorFlow 2.19
h5py · nibabel · scikit-learn · scipy · matplotlib · numpy
```

```bash
pip install -r requirements.txt
```

---

## Training

The published metrics come from **`glioma_colab_final_v2.ipynb`** (binary) and **`glioma_multiclass_colab.ipynb`** (multiclass), both trained on a Tesla T4 with batch size 16. `glioma.py` is the same pipeline adapted to a local machine with less GPU memory: it uses batch size 8, so its results are not expected to match the reported metrics exactly.

### Option 1 — Google Colab (reproduces the paper)
1. Open `glioma_colab_final_v2.ipynb` in Google Colab
2. Set the runtime to **T4 GPU**
3. Upload your `kaggle.json` when the notebook asks for it
4. Run the cells in order

### Option 2 — Local (Windows/Linux)
Edit `DATA_DIR` in `glioma.py` to point to your BraTS2020 folder, then:

```bash
pip install -r requirements.txt
python glioma.py
```

---

## Model and training setup

2D U-Net encoder-decoder:
- Encoder: 32 → 64 → 128 → 256 filters with max pooling
- Bottleneck: 512 filters
- Decoder: transposed convolutions with skip connections
- Output: sigmoid (binary) or per-class sigmoid (multiclass)
- Input: 128×128×4 · ~7.8M trainable parameters
- Callbacks: ModelCheckpoint (best val DSC), ReduceLROnPlateau, EarlyStopping

| Setting | Notebooks (paper) | `glioma.py` (local) |
|---------|-------------------|---------------------|
| GPU | NVIDIA Tesla T4, 15 GB (Google Colab) | consumer GPU, <8 GB |
| Framework | TensorFlow 2.19 | TensorFlow 2.19 |
| Optimizer | Adam, lr = 1e-3 | Adam, lr = 1e-3 |
| Loss | Dice loss | Dice loss |
| Input size | 128×128×4 | 128×128×4 |
| Batch size | 16 | 8 |
| Epochs | 15 | 15 |
| Seed | 42 | 42 |

---

## Evaluation metrics

Dice Similarity Coefficient (DSC), Sensitivity, Specificity and Hausdorff Distance. Metrics are computed per slice and averaged. The Hausdorff distance is computed with `scipy.spatial.distance.directed_hausdorff` over pixel coordinates of the resized 128×128 grid; no voxel-spacing conversion is applied in the code. Predictions are binarized with a fixed threshold of 0.5. Slices where both the ground truth and the prediction are empty are excluded from the Hausdorff average.

---

## Limitations

- Evaluated only on BraTS2020; no validation on clinical data from hospitals.
- The split is by slice, not by patient, so the reported metrics may be optimistic relative to a patient-level split.
- The 2D slice-based approach discards inter-slice context, which can affect tumors with complex 3D morphology.
- The binarization threshold is fixed and may not be optimal for every case.
- No direct comparison against 3D or attention-based architectures under identical conditions.

---

## Citation

```bibtex
@inproceedings{reyeshung2026glioma,
  author    = {Reyes-Hung, L. and Trinke, G. and Soto, I. and Serey, J.},
  title     = {Optimizing Magnetic Resonance Image Segmentation Through Scalable Deep Learning and Hierarchical Data Management},
  booktitle = {2026 15th International Symposium on Communication Systems, Networks and Digital Signal Processing (CSNDSP)},
  address   = {Edinburgh, United Kingdom},
  year      = {2026},
  pages     = {1--6},
  doi       = {10.1109/CSNDSP68462.2026.11654508}
}
```

---

## Acknowledgments

Supported by FONDECYT Regular (Grant No. 1261732/2026), ANID/FIU (Grant No. 137139) and ANID BECAS/DOCTORADO NACIONAL 21242235, Vicerrectoría de Postgrado, Universidad de Santiago de Chile, through Becas 2026.
