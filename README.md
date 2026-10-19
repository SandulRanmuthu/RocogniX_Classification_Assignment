# RecogniX – Resource-Constrained CNN for Edge Image Classification

EN3150 Pattern Recognition – Assignment 03

Head-pose classification (**left, right, straight, up**) on the CMU Face Images dataset with small CNNs designed for edge devices. Two custom CNNs are compared with two fine-tuned lightweight pretrained networks.

## Group members

| Name | Index No. |
|---|---|
| Ranmuthu S.G. | 230532M |
| Perera K.S.D. | 230480B |
| Suraweera B.U.D. | 230626F |
| Gunawardena H.A. | 230219K |

## Technical overview

- **Framework:** TensorFlow / Keras 3, trained on CPU.
- **Input:** 64×60 grayscale images. For the custom models, pixels are scaled to [0, 1] and standardised with the training-set mean and standard deviation. For the pretrained models, the grayscale channel is repeated to 3 channels and rescaled to each network's expected input range.
- **Custom models:** both use 3×3 convolutions, each followed by batch normalisation and ReLU, with He initialisation, 2×2 max pooling and global average pooling instead of flatten. Model B replaces every standard convolution with a depthwise separable one (3×3 depthwise + 1×1 pointwise) to cut MACs by about 7.5×. ReLU is used because it needs no exponentials and is cheap on edge hardware.
- **Training:** Adam (lr = 1e-3), batch size 32 and sparse categorical cross-entropy. Model A was trained for 100 epochs with `ReduceLROnPlateau`, and Model B for 50 epochs.
- **Optimizer study:** Adam was compared with SGD (lr = 0.05) and SGD with momentum 0.9 (lr = 0.01) from identical initial weights. A momentum sweep (0, 0.5, 0.9, 0.99) was also run on Model A.
- **Transfer learning:** MobileNetV2 and EfficientNet-B0 use ImageNet weights and a new head (GAP -> Dense(128) -> Dropout(0.3) -> Dense(4)). They are trained in two stages: 20 epochs with a frozen backbone (lr = 1e-3), then 20 epochs fine-tuning the last 30 layers (lr = 1e-5, BatchNorm kept frozen).
- **Evaluation:** test accuracy, confusion matrix, per-class precision and recall, macro F1, parameter count, MACs, model size and training time per epoch.

## Dataset

- **Source:** [CMU Face Images](https://archive.ics.uci.edu/dataset/124/cmu+face+images), UCI Machine Learning Repository.
- **Images:** the 64×60 grayscale version (`halfres`), already within the 64×64 limit. Corrupted `.bad` files were removed, leaving 624 images.
- **Label:** the pose, read from the file name (`userid_pose_expression_eyes_scale.pgm`).
- **Split:** random 70/15/15 split with seed 42 (`FaceData_Raw/split.py`). The same subjects appear in all three splits.

## Models

| Model | Description | Total params | MACs / image |
|---|---|---|---|
| Model A | 4× Conv2D (32-64-128-128) + BN + ReLU, max pooling, GAP, Dense(64) | 249,828 | 44.8 M |
| Model B | 4× SeparableConv2D (32-64-128-192) + BN + ReLU, max pooling, GAP, Dense(128) | 63,757 (62,925 trainable) | 5.97 M |
| MobileNetV2 | ImageNet weights, new head, two-stage fine-tuning | 2,422,468 | 24.2 M |
| EfficientNet-B0 | ImageNet weights, new head, two-stage fine-tuning | 4,214,055 | 31.7 M |

## Results (test set, 95 images)

| Model | Test accuracy | Macro F1 | Size (FP32) |
|---|---|---|---|
| Model A | 98.95% | 0.9896 | 0.95 MB |
| Model B | 94.74% | 0.9470 | 0.24 MB |
| MobileNetV2 | 98.95% | 0.9896 | 9.24 MB |
| EfficientNet-B0 | 100.00% | 1.0000 | 16.08 MB |

All models use Adam (lr = 1e-3) as the main optimizer. The full analysis, including the comparison of Adam, SGD and SGD with momentum, is in [`Recognix_A03_EN3150.pdf`](Recognix_A03_EN3150.pdf).

## How to run

1. Install the dependencies (Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Re-create the split. Set `SOURCE_DIR` and `OUTPUT_DIR` at the top of `FaceData_Raw/split.py` to your local paths, then run:
   ```bash
   python FaceData_Raw/split.py
   ```
   The ready-made split is already included in `split/`.
3. Open and run the notebooks from the repository root (they read images from `split/`):
   ```bash
   jupyter notebook
   ```
   Run `modelA.ipynb`, `modelB.ipynb`, `MobileNetV2.ipynb` and `EfficientNet-B0.ipynb` top to bottom.

All random seeds are fixed to 42. Training was done on CPU (TensorFlow 2.21 on Windows), so timings and exact results may differ slightly on other hardware.
