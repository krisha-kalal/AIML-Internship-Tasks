# Intel Scene Image Classification: Custom CNN vs. ResNet-18 Transfer Learning

An end-to-end PyTorch multiclass deep learning pipeline designed to classify natural and urban scene images into six distinct categories (`buildings`, `forest`, `glacier`, `mountain`, `sea`, and `street`) using the [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

This project implements, evaluates, and compares two distinct computer vision approaches:
1. **Model 1 (`SceneCNN`)**: A custom 4-block Convolutional Neural Network built from scratch using `torch.nn.Module`.
2. **Model 2 (`ResNet-18`)**: A pretrained `ResNet18_Weights.DEFAULT` architecture trained via a two-stage transfer learning workflow (**Stage 1: Feature Extraction** with a frozen backbone, followed by **Stage 2: Fine-Tuning** of `layer4` and the classification head).

---

## 📁 Project Structure

```text
intel-image-classifier/
├── data/                         # Dataset directory (not tracked in repository)
│   ├── seg_train/seg_train/      # Labeled training images (split 85% train / 15% val)
│   ├── seg_test/seg_test/        # Untouched labeled test images for final evaluation
│   └── seg_pred/seg_pred/        # Unlabeled images for inference demonstration
├── notebooks/
│   └── exploration.ipynb         # EDA, integrity checks, class distributions & augmentation grids
├── src/
│   ├── preprocessing.py          # Training augmentations & deterministic Val/Test transforms
│   ├── data_loader.py            # Leakage-free seeded splits & PyTorch DataLoaders
│   ├── cnn_model.py              # Custom 4-block CNN architecture & parameter counter
│   ├── resnet_model.py           # Pretrained ResNet-18 builder & layer freeze/unfreeze utilities
│   ├── train.py                  # Training & validation workflow for Custom CNN
│   ├── train_resnet.ipynb        # Two-stage training workflow for ResNet-18
│   ├── evaluate.ipynb            # Final test-set evaluation, confusion matrices & error analysis
│   └── predict.ipynb             # Reusable single-image & batch inference pipeline
├── models/
│   ├── cnn_best.pth              # Best Custom CNN checkpoint (selected via validation accuracy)
│   └── resnet18_best.pth         # Best ResNet-18 checkpoint (selected via validation accuracy)
├── outputs/
│   ├── plots/                    # Loss/accuracy curves, misclassified grids & inference demos
│   ├── metrics/                  # Epoch-wise JSON logs, test reports & experiment CSVs
│   └── confusion_matrices/       # Confusion matrix heatmaps for both models
├── requirements.txt              # Python package dependencies
└── README.md                     # Project overview, setup & execution instructions
```

---

## Setup & Installation

### 1. Clone or Navigate to the Project Directory
```bash
cd intel-image-classifier
```

### 2. Install Dependencies
Ensure you have Python 3.9+ and a CUDA-enabled GPU environment (such as Google Colab or Kaggle Notebooks) recommended for training:
```bash
pip install -r requirements.txt
```

### 3. Dataset Preparation
Download the dataset from Kaggle (`puneet6060/intel-image-classification`) and extract it into the `data/` directory so the folder hierarchy matches:
* `data/seg_train/seg_train/<class_name>/*.jpg`
* `data/seg_test/seg_test/<class_name>/*.jpg`
* `data/seg_pred/seg_pred/*.jpg`

---

## Execution Workflow

Run the modules in the following sequence to reproduce all training runs, experiments, evaluations, and inference outputs:

### Step 1: Verify Model Architectures & Parameter Counts
Inspect layer dimensions and verify trainable vs. frozen parameter counts before training:
```bash
python src/cnn_model.py
python src/resnet_model.py
```

### Step 2: Train Custom CNN from Scratch (Model 1)
Trains `SceneCNN` on $150 \times 150$ RGB tensors using Cross-Entropy Loss and the Adam optimizer (`lr=1e-3`, `weight_decay=1e-4`), saving the best checkpoint to `models/cnn_best.pth`:
```bash
python src/train.py
```

### Step 3: Train ResNet-18 via Transfer Learning (Model 2)
Trains `ResNet-18` on $224 \times 224$ RGB tensors in two stages—first training only the newly attached 6-class linear head (`lr=1e-3`), then unfreezing `layer4` for fine-tuning (`lr=1e-4`)—saving the best checkpoint to `models/resnet18_best.pth`:
```bash
python src/train_resnet.py
```

### Step 4: Final Evaluation & Error Analysis (Untouched Test Set)
Evaluates both `cnn_best.pth` and `resnet18_best.pth` once on `seg_test`, generating classification reports (Accuracy, Macro/Weighted Precision, Recall, and F1-Score), confusion matrices, and visual grids of the top 10 misclassified test images:
```bash
python src/evaluate.py
```

### Step 5: Run Inference on Unlabeled Images
Executes `predict_image()` with `torch.no_grad()` on external/unlabeled images from `seg_pred`, returning predicted classes and softmax confidence scores across all six categories:
```bash
python src/predict.py
```

---

## Methodology & Design Decisions

* **Data Leakage Prevention**: The original training directory (`seg_train`) is split 85% / 15% using a fixed generator seed (`seed=42`). Custom dataset wrappers apply stochastic augmentations (`RandomHorizontalFlip`, `RandomRotation(15)`, `ColorJitter`) **exclusively** to the training split, while validation and test sets use strictly deterministic resizing and ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
* **Test Set Integrity**: All architecture choices, learning rate adjustments, and checkpoint selections rely strictly on validation performance. The labeled `seg_test` set is reserved solely for final evaluation.
* **Reproducibility**: Random seeds are fixed across PyTorch CPU/CUDA operations (`seed=42`), and saved `.pth` checkpoints bundle model state dictionaries alongside `class_names` and input resolution metadata to ensure consistent inference.