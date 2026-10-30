# From-Scratch Face Recognition Attendance System

This repository contains the code for a custom Face Recognition Attendance System. The system is built entirely from scratch (random weight initialization, no pre-trained models). It uses a custom Convolutional Neural Network (CNN) trained with **ArcFace Loss**, and uses **Cosine Similarity** for local identification and unknown-face rejection.

## 📂 Project Structure

```text
├── train_model.ipynb             # Notebook for training the model from scratch (Kaggle)
├── test_inference.py             # Script for local enrollment, evaluation, and attendance logging
├── arcface_public_trained_cnn.pth# Final trained PyTorch weights (Included in repo)
├── README.md                     # Project documentation
└── TRAINING PICS/                # Local directory for user enrollment
    ├── Akira/
    ├── Emman/
    ├── Louie/
    └── Ronald/
```

## 📊 Datasets

### 1. Public Training Dataset (Cloud)
*   **Dataset:** [Prosopo Small Dataset (Aligned CASIA-WebFace)](https://www.kaggle.com/datasets/inanxr/prosopo-small-dataset)
*   **Details:** ~494,000 images across 9,934 identities, pre-cropped to 112x112 pixels. Used to teach the CNN general facial geometry from scratch.

### 2. Local Enrollment Dataset (Private)
*   **Format:** A folder named `TRAINING PICS` containing subfolders named after each person (`Akira`, `Emman`, `Louie`, `Ronald`).
*   **Contents:** 2 to 9 images per person (JPEG, PNG, or HEIC).

---

## 🛠️ Environment Setup

To run the inference script locally, install the required Python libraries:

```bash
pip install torch torchvision torchaudio
pip install facenet-pytorch
pip install pillow pillow-heif numpy
```
*(Note: `pillow-heif` is required to process Apple `.HEIC` images).*

---

## 🚀 Reproduction Instructions

### Option 1: Quick Start (Inference Only)
If you want to test the system immediately without re-training the model, the pre-trained weights (`arcface_public_trained_cnn.pth`) are already included in this repository.

1. Clone this repository.
2. Ensure the `TRAINING PICS` folder contains the images you want to test.
3. Run the script:
```bash
python test_inference.py
```

### Option 2: Full Reproduction (Training from Scratch)
Because training a CNN from scratch requires significant compute, this is done on a free Kaggle GPU.

**Phase 1: Training the Model**
1. Open `train_model.ipynb` in **Kaggle**.
2. In the Notebook settings, set the **Accelerator** to `GPU T4 x2` or `GPU P100`.
3. Add the Prosopo Small Dataset to the notebook via the "Add Data" button.
4. Run all cells. The script is configured to train for 15 epochs. To bypass Kaggle disk I/O bottlenecks, the dataloader uses `max_batches=200` per epoch. The model weights are saved to the output folder after every epoch.
5. Download the final `arcface_public_trained_cnn.pth` file from the Kaggle output directory and place it in the root of your local project.

**Phase 2: Local Evaluation**
1. Place `arcface_public_trained_cnn.pth`, `test_inference.py`, and the `TRAINING PICS` folder in the same local directory.
2. Run `python test_inference.py`.

**Execution Logic:**
*   The script utilizes MTCNN to detect and align faces.
*   **Leave-One-Out Evaluation:** For each person's folder, it automatically holds out the last image for testing.
*   **Enrollment:** It extracts 128-D embeddings for the remaining images, averages them into a single prototype vector, and L2-normalizes it.
*   **Inference:** It calculates the Cosine Similarity between the held-out image and all prototypes. If the highest similarity exceeds the `threshold` (default `0.65`), attendance is logged; otherwise, it returns "Unknown".

---

## 🧠 Architecture & Methodology

*   **CNN Architecture (`FaceEmbeddingCNN`):** A custom 3-layer convolutional network. Each block consists of `Conv2d` -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d`. The output is flattened and passed through a `Linear` layer to generate a 128-dimensional embedding, which is then L2-normalized.
*   **ArcFace Loss:** Implemented to maximize intra-class similarity and minimize inter-class similarity. 
*   **Stability Fixes:** To prevent "Network Collapse" (dead ReLUs caused by exploding gradients), `BatchNorm2d` was introduced, the learning rate was reduced to `0.0001`, and ArcFace parameters were softened (`margin=0.3`, `scale=16.0`).