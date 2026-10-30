Here is the complete, updated `README.md`. 

I have given `test_inference.py` its own dedicated section just like the live prototype, and removed the markdown code blocks so you can copy and paste this directly into your file.

# From-Scratch Face Recognition Attendance System

This repository contains the code for a custom Face Recognition Attendance System. The system is built entirely from scratch (random weight initialization, no pre-trained models). It uses a custom Convolutional Neural Network (CNN) trained with **ArcFace Loss**, and uses **Cosine Similarity** for local identification and unknown-face rejection.

## 📂 Project Structure

```text
├── train_model.ipynb                     # Notebook for training the model from scratch (Kaggle)
├── test_inference.py                     # Script for automated local evaluation (Leave-one-out)
├── prototype_Facial_Attendance_System.py # Live webcam attendance & auto-enrollment script
├── arcface_public_trained_cnn.pth        # Final trained PyTorch weights (Included in repo)
├── attendance.csv                        # Generated log file for attendance tracking
├── README.md                             # Project documentation
└── TRAINING PICS/                        # Local directory for user enrollment
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

To run the scripts locally, install the required Python libraries:

```bash
pip install torch torchvision torchaudio
pip install facenet-pytorch
pip install pillow pillow-heif numpy
pip install opencv-python keyboard
```
*(Note: `pillow-heif` is required to process Apple `.HEIC` images. `opencv-python` and `keyboard` are required for the live camera prototype).*

---

## 🚀 Reproduction Instructions

### Option 1: Quick Start (Inference Only)
If you want to test the system immediately without re-training the model, the pre-trained weights (`arcface_public_trained_cnn.pth`) are already included in this repository.

1. Clone this repository.
2. Ensure the `TRAINING PICS` folder contains the images you want to test.
3. To run the automated evaluation, run: `python test_inference.py`
4. To run the live webcam attendance prototype, run: `python prototype_Facial_Attendance_System.py`

### Option 2: Full Reproduction (Training from Scratch)
Because training a CNN from scratch requires significant compute, this is done on a free Kaggle GPU.

**Phase 1: Training the Model**
1. Open `train_model.ipynb` in **Kaggle**.
2. In the Notebook settings, set the **Accelerator** to `GPU T4 x2` or `GPU P100`.
3. Add the Prosopo Small Dataset to the notebook via the "Add Data" button.
4. Run all cells. The script is configured to train for 15 epochs. To bypass Kaggle disk I/O bottlenecks, the dataloader uses `max_batches=200` per epoch. The model weights are saved to the output folder after every epoch.
5. Download the final `arcface_public_trained_cnn.pth` file from the Kaggle output directory and place it in the root of your local project.

**Phase 2: Local Evaluation**
1. Place `arcface_public_trained_cnn.pth`, the Python scripts, and the `TRAINING PICS` folder in the same local directory.
2. Run the desired script.

---

## 🧪 Automated Test Script (`test_inference.py`)

This script is designed specifically for benchmarking and generating results for academic evaluation. It performs a strict **Leave-One-Out cross-validation** to ensure the model is not just memorizing training data.

*   **Leave-One-Out Evaluation:** For each person's folder, it automatically holds out the last image for testing and ignores it during the enrollment phase.
*   **Enrollment:** It extracts 128-D embeddings for the remaining images in the folder, averages them into a single prototype vector, and L2-normalizes it.
*   **Inference:** It calculates the Cosine Similarity between the held-out image and all prototypes. If the highest similarity exceeds the `threshold` (default `0.65`), it counts as a correct match.
*   **Output:** Generates a formatted terminal table showing the true identity, test image filename, predicted identity, similarity score, and the final overall accuracy percentage.

---

## 🎥 Live Camera Prototype (`prototype_Facial_Attendance_System.py`)

This script opens your default webcam and provides real-time attendance tracking. Unlike the test script, it uses **all** available images in the `TRAINING PICS` folder to build the strongest possible facial profiles for live use.

*   **Global Hotkeys:** Works reliably without needing to click the window.
    *   `q`: Quit the application.
    *   `s`: Save a snapshot of a recognized face to instantly update their database profile.
    *   `e`: Enroll an unknown face.
*   **Auto-Enrollment:** When an unknown face is detected, press `e`. The system will recommend the closest matching identity (if any). You can type a new name to create a folder in `TRAINING PICS` and save the 112x112 cropped face instantly.
*   **Live Memory Update:** When you press `s` to save a snapshot, the system calculates the new running average immediately in RAM, so the model gets smarter instantly without needing a restart.
*   **Smart Cooldown:** To prevent spamming the CSV, the 30-second cooldown timer only starts after the person leaves the camera frame.
*   **CSV Logging:** Attendance is logged in `attendance.csv` with the format: `Timestamp, Name`.

---

## 🧠 Architecture & Methodology

*   **CNN Architecture (`FaceEmbeddingCNN`):** A custom 3-layer convolutional network. Each block consists of `Conv2d` -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d`. The output is flattened and passed through a `Linear` layer to generate a 128-dimensional embedding, which is then L2-normalized.
*   **ArcFace Loss:** Implemented to maximize intra-class similarity and minimize inter-class similarity. 
*   **Stability Fixes:** To prevent "Network Collapse" (dead ReLUs caused by exploding gradients), `BatchNorm2d` was introduced, the learning rate was reduced to `0.0001`, and ArcFace parameters were softened (`margin=0.3`, `scale=16.0`).