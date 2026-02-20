# Pedestrian Intention Prediction for Autonomous Vehicles

A deep learning system that predicts pedestrian crossing intentions using YOLO object detection and LSTM temporal modeling. This project is designed for autonomous vehicle applications to enhance safety by predicting pedestrian behavior before they enter the roadway.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Computer Vision](https://img.shields.io/badge/Field-Computer%20Vision-blue.svg)](#)
[![LSTM](https://img.shields.io/badge/LSTM-Keras-red?logo=keras&logoColor=white)](https://en.wikipedia.org/wiki/Long_short-term_memory)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-green.svg)](https://ultralytics.com/)
[![Temporal Modeling](https://img.shields.io/badge/Temporal-Sequence%20Learning-pink.svg)](#)
[![Behavior Modeling](https://img.shields.io/badge/Behavior-Temporal%20Prediction-purple.svg)](#)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg)](https://www.kaggle.com/code/kimiakarbasi/pedestrian-intuition-ue/notebook)
[![Dataset](https://img.shields.io/badge/Dataset-JAAD-darkgreen.svg)](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[View on Kaggle](https://www.kaggle.com/code/kimiakarbasi/pedestrian-intuition-ue/notebook)**

---

## Quick Start (Kaggle)

1. **Open the notebook**: [Click here](https://www.kaggle.com/code/kimiakarbasi/pedestrian-intuition-ue/notebook)
2. **Add datasets** (see [Dataset section](#dataset) for details):
   - `vehic-ped-intuition` (required)
   - `attributes-label` (required)
   - `first-phase-model` (required)
3. **Enable GPU** in notebook settings (T4 x2 or higher)
4. **Run all cells** sequentially

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Performance](#performance)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Results](#results)
- [Requirements](#requirements)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [Disclaimer](#disclaimer)
- [Citation](#citation)
- [License](#license)




---

## Overview

This project implements a two-stage pipeline for pedestrian intention prediction:

1. **Detection & Tracking**: YOLOv8-based pedestrian detection with persistent tracking across video frames
2. **Intention Prediction**: LSTM neural network that analyzes temporal sequences of pedestrian movement features to predict crossing intentions

The system processes video frames in real-time, tracks pedestrians, extracts kinematic features, and predicts whether a pedestrian will **CROSS** the road or **STAY** on the sidewalk.

### Key Innovation

- **CLAHE Enhancement**: Low-light image enhancement using Contrast Limited Adaptive Histogram Equalization (CLAHE) in LAB color space
- **Temporal Feature Engineering**: Extracts velocity, position, scale change, and aspect ratio from tracked trajectories
- **Before Decision Point Learning**: Trains on frames before the pedestrian's decision point to enable early prediction

---

## Features

- **Real-time Pedestrian Detection**: YOLOv8-based detection with BoT-SORT tracking
- **Low-light Enhancement**: CLAHE preprocessing for improved detection in dark/night scenes
- **Temporal Modeling**: LSTM architecture for sequence-based intention prediction
- **XML Metadata Integration**: Handshaking between YOLO track IDs and ground truth pedestrian IDs
- **Class Balancing**: Oversampling techniques to handle imbalanced datasets
- **Video-level Data Splitting**: Ensures no data leakage between train/val/test sets
- **Comprehensive Evaluation**: Confusion matrices, classification reports, and visualizations
- **Deployment Ready**: Serialized models and scalers for production use

---

## Architecture

### System Pipeline


| Stage | Component | Input | Output |
|-------|-----------|-------|--------|
| 1 | **CLAHE Enhancement** | Video Frames | Improved visibility in low-light |
| 2 | **YOLO Detection + Tracking** | Enhanced Frames | Pedestrian bounding boxes + Track IDs |
| 3 | **Feature Extraction** | Tracked Pedestrians | x, y, vel_x, vel_y, delta_area, aspect_ratio |
| 4 | **Sequence Building** | Extracted Features | Sliding windows of 10 frames |
| 5 | **LSTM Prediction** | Feature Sequences | Crossing probability (0-1) |
| 6 | **Visualization** | Predictions | Annotated video with predictions |


### Model Architecture

#### YOLO Detector
- **Model**: YOLOv8n (nano)
- **Input**: 640×640 RGB images
- **Output**: Bounding boxes + confidence scores
- **Classes**: Pedestrian (class 0)
- **Tracker**: BoT-SORT with `persist=True`

#### LSTM Predictor
- **Architecture**: 2-layer LSTM
- **Input Size**: 6 features per frame
- **Hidden Size**: 64 units
- **Sequence Length**: 10 frames
- **Output**: Binary classification (Crossing/Staying)

### Feature Engineering

The system extracts 6 kinematic features from each tracked pedestrian:

| Feature | Description |
|---------|-------------|
| `x, y` | Center coordinates of bounding box |
| `vel_x, vel_y` | Velocity components (movement between frames) |
| `delta_area` | Scale change ratio (>1.0 = approaching camera) |
| `aspect_ratio` | Width/Height ratio (posture indicator) |

---

## Performance

### Model Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Overall Accuracy** | 84% | High reliability across dataset |
| **Crossing Recall** | 86% | Successfully identifies most crossing events |
| **Crossing Precision** | 96% | Very low false alarm rate |
| **Staying Recall** | 60% | Captures majority of minority class |

### Training Details

- **Epochs**: 40 (with early stopping, patience=15)
- **Batch Size**: 64
- **Optimizer**: Adam (lr=0.0005, weight_decay=1e-3)
- **Loss Function**: BCEWithLogitsLoss with class weights
- **Best Model**: Saved at epoch 4 (lowest validation loss)

---

## Installation

### Option 1: Kaggle Notebook (Recommended)

This project is designed to run on **Kaggle Notebooks** with GPU acceleration.

#### Prerequisites
- Kaggle account (free)
- Enable GPU accelerator in Kaggle settings

#### Step-by-Step Setup on Kaggle

1. **Open the Notebook**
   - Go to: [Pedestrian Intuition - Kaggle Notebook](https://www.kaggle.com/code/kimiakarbasi/pedestrian-intuition-ue/notebook)
   - Click "Copy & Edit" to create your own copy

2. **Add Required Kaggle Datasets**
   
   Add these datasets to your Kaggle notebook inputs:
   
   | Dataset Name | Description | Path in Notebook |
   |-------------|-------------|------------------|
   | `vehic-ped-intuition` | Main dataset with images and labels | `/kaggle/input/vehic-ped-intuition` |
   | `attributes-label` | XML annotations with pedestrian attributes | `/kaggle/input/attributes-label` |
   | `first-phase-model` | Pre-trained YOLO model weights | `/kaggle/input/first-phase-model` |
   | `phase-3-dataset` | Pre-processed sequences (optional) | `/kaggle/input/phase-3-dataset` |
   | `phase-3-lstm-yolo` | Trained LSTM models (optional) | `/kaggle/input/phase-3-lstm-yolo` |

3. **Enable GPU**
   - In notebook settings, select **GPU T4 x2** or higher
   - This is essential for YOLO training and LSTM inference

4. **Run the Notebook**
   - The notebook will automatically install dependencies (`ultralytics`, `torch`)
   - All paths are pre-configured for Kaggle environment
   - Outputs are saved to `/kaggle/working/`

#### Kaggle Directory Structure

```
/kaggle/
├── input/
│   ├── vehic-ped-intuition/          # Main dataset
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── labels/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   ├── attributes-label/               # XML metadata
│   │   └── annotations_attributes/
│   ├── first-phase-model/             # YOLO weights
│   │   └── weights/
│   │       └── best.pt
│   ├── phase-3-dataset/               # Pre-processed data (optional)
│   └── phase-3-lstm-yolo/             # LSTM models (optional)
│       └── Phase_3_models/
│           └── lstm_intention_model.pth
│
└── working/                            # Output directory
    ├── data.yaml                       # YOLO config
    ├── master_*_dataset.csv            # Feature datasets
    ├── balanced_train_dataset.csv      # Balanced training data
    ├── X_train.npy, X_val.npy, X_test.npy  # Sequences
    ├── training_results/               # YOLO training outputs
    ├── evaluation/                     # Evaluation results
    └── final_results/                  # Final videos
```

### Option 2: Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/pedestrian-intention-prediction.git
cd pedestrian-intention-prediction
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download datasets** and update paths in the notebook accordingly

---

## Dataset

---

### Raw Dataset (Original JAAD Data)

If you want to work with the original raw videos and annotations — without pre-generated feature files or processed frame splits — you can download the official JAAD dataset here:

[JAAD Dataset](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/)

**Includes:**
- Full raw driving videos
- XML pedestrian annotations
- Behavioral attributes
- Crossing intention labels

Recommended if you want to:
- Reproduce the full preprocessing pipeline
- Generate custom frame extraction
- Modify tracking or feature extraction stages
- Extend the dataset with additional temporal features

### Preprocessed Kaggle Datasets

This project requires the following datasets to be added to your Kaggle notebook:

#### 1. Main Dataset: `vehic-ped-intuition`
- **Path**: `/kaggle/input/vehic-ped-intuition`
- **Contents**:
  - `images/train/`, `images/val/`, `images/test/` - Video frames (JPG)
  - `labels/train/`, `labels/val/`, `labels/test/` - YOLO format annotations
- **Format**: Frames named as `video_XXXX_frame_YYYY.jpg`
- **Labels**: YOLO format `(class, x_center, y_center, width, height)` normalized to [0,1]

#### 2. Attributes Dataset: `attributes-label`
- **Path**: `/kaggle/input/attributes-label`
- **Contents**: `annotations_attributes/video_XXXX_attributes.xml`
- **XML Format**: Contains pedestrian metadata:
  ```xml
  <pedestrian id="X" decision_point="Y" crossing="0/1">
  ```
  - `decision_point`: Frame number when pedestrian makes decision
  - `crossing`: Binary label (0=stay, 1=cross)

#### 3. Pre-trained YOLO Model: `first-phase-model`
- **Path**: `/kaggle/input/first-phase-model/weights/best.pt`
- **Description**: Trained YOLOv8n model for pedestrian detection
- **Usage**: Used for Phase 2 tracking and feature extraction

#### 4. Optional: Pre-processed Data (`phase-3-dataset`)
- **Path**: `/kaggle/input/phase-3-dataset`
- **Contents**: Pre-generated sequences (`X_train.npy`, `X_val.npy`, `X_test.npy`)
- **Note**: Can be generated by running the notebook from scratch

#### 5. Optional: Trained LSTM Models (`phase-3-lstm-yolo`)
- **Path**: `/kaggle/input/phase-3-lstm-yolo/Phase_3_models/lstm_intention_model.pth`
- **Description**: Pre-trained LSTM model weights
- **Note**: Can be trained by running the notebook

### Dataset Structure (Kaggle)

```
/kaggle/input/vehic-ped-intuition/
├── images/
│   ├── train/          # Training frames
│   ├── val/            # Validation frames
│   └── test/           # Test frames
├── labels/
│   ├── train/          # YOLO labels for training
│   ├── val/            # YOLO labels for validation
│   └── test/           # YOLO labels for test
└── crops/              # Optional: Pre-cropped pedestrian images

/kaggle/input/attributes-label/
└── annotations_attributes/
    └── video_XXXX_attributes.xml  # Pedestrian metadata
```

### Data Preprocessing Pipeline

The notebook includes automatic preprocessing:

1. **CLAHE Enhancement**: Applied to all frames before detection (improves dark scene visibility)
2. **Video-level Splitting**: Ensures no video appears in multiple splits (prevents data leakage)
3. **YOLO Tracking**: Persistent tracking with `persist=True` maintains IDs across frames
4. **XML Handshaking**: Matches YOLO track IDs with ground truth pedestrian IDs
5. **Feature Extraction**: Computes 6 kinematic features per frame
6. **Sequence Filtering**: Only sequences with ≥5 consecutive frames are kept
7. **Class Balancing**: Oversampling applied to training set (50/50 split)
8. **Sequence Building**: Creates sliding windows of 10 frames with StandardScaler normalization

---

## Usage

### Running on Kaggle

The notebook is organized into sequential cells. Simply run all cells in order:

1. **Phase 1: YOLO Training** 
   - Dataset discovery
   - YOLO installation and GPU verification
   - Model training (50 epochs)
   - Evaluation on test set

2. **Phase 2: Feature Extraction** 
   - CLAHE enhancement implementation
   - Multivideo processor for train/val/test splits
   - Feature extraction and CSV generation
   - Data balancing

3. **Phase 3: LSTM Training** 
   - Sequence building
   - LSTM model definition
   - Training loop with early stopping
   - Evaluation and visualization

4. **Phase 4: Real-time Inference** 
   - Model loading
   - Video processing pipeline
   - Visualization and output generation

### Key Code Snippets

#### 1. YOLO Training (Kaggle Paths)

```python
from ultralytics import YOLO
import yaml

# Kaggle paths
DATASET_ROOT = '/kaggle/input/vehic-ped-intuition'
WORKING_DIR = '/kaggle/working'

# Create data.yaml configuration
data_config = {
    'train': f'{DATASET_ROOT}/images/train',
    'val': f'{DATASET_ROOT}/images/val',
    'test': f'{DATASET_ROOT}/images/test',
    'nc': 1,
    'names': ['pedestrian']
}

with open(f'{WORKING_DIR}/data.yaml', 'w') as f:
    yaml.dump(data_config, f)

# Train YOLO model
model = YOLO('yolov8n.pt')
model.train(
    data=f'{WORKING_DIR}/data.yaml',
    epochs=50,
    imgsz=640,
    batch=32,
    device=0,
    name='jaad_final_model',
    project=f'{WORKING_DIR}/training_results'
)
```

#### 2. Feature Extraction (Multivideo Processor)

```python
# Kaggle paths
BASE_PATH = '/kaggle/input/vehic-ped-intuition'
XML_PATH = '/kaggle/input/attributes-label/annotations_attributes'
MODEL_PATH = '/kaggle/input/first-phase-model/weights/best.pt'

# Process all splits
for split in ['train', 'val', 'test']:
    process_split(split)
    
# Outputs saved to /kaggle/working/:
# - master_train_dataset.csv
# - master_val_dataset.csv
# - master_test_dataset.csv
```

#### 3. LSTM Training

```python
# Load sequences
X_train = np.load('/kaggle/working/X_train.npy')
y_train = np.load('/kaggle/working/y_train.npy')
X_val = np.load('/kaggle/working/X_val.npy')
y_val = np.load('/kaggle/working/y_val.npy')

# Train model
model = IntentionLSTM(input_size=6, hidden_size=64, num_layers=2).to(device)
# ... training loop ...

# Save model
torch.save(model.state_dict(), '/kaggle/working/lstm_intention_model.pth')
```

#### 4. Real-time Inference

```python
# Load models from Kaggle inputs
yolo_model = YOLO('/kaggle/input/first-phase-model/weights/best.pt')
lstm_model = torch.load('/kaggle/input/phase-3-lstm-yolo/Phase_3_models/lstm_intention_model.pth')
scaler = joblib.load('/kaggle/input/phase-3-dataset/data_scaler.pkl')

# Process video
process_video_realtime(
    folder_path='/kaggle/input/vehic-ped-intuition/images/test',
    video_id='video_0039',
    output_path='/kaggle/working/final_results/result_video_0039.mp4',
    yolo_model=yolo_model,
    lstm_model=lstm_model,
    scaler=scaler
)
```

---

## Project Structure

### Kaggle Notebook Structure

The project is organized as a single Kaggle notebook with the following structure:

```
Kaggle Notebook: Pedestrian-Intuition.ipynb
│
├── Phase 1: YOLO Detection 
│   ├── Dataset Discovery
│   ├── YOLO Installation & Setup
│   ├── GPU Verification
│   ├── Model Training (50 epochs)
│   ├── Test Set Evaluation
│   └── Video-level Leakage Check
│
├── Phase 2: Data Preparation 
│   ├── CLAHE Enhancement Implementation
│   ├── Tracking + XML Handshaking
│   ├── Multivideo Processor
│   ├── Feature Extraction
│   └── Data Balancing
│
├── Phase 3: LSTM Training 
│   ├── Sequence Building
│   ├── LSTM Architecture Definition
│   ├── Training Loop
│   ├── Evaluation Metrics
│   └── Visualization
│
└── Phase 4: Real-time Inference (Cells 31-35)
    ├── Model Loading
    ├── Video Processing Pipeline
    └── Output Generation
```

### Kaggle Directory Structure

```
/kaggle/
│
├── input/                              # Read-only datasets
│   ├── vehic-ped-intuition/           # Main dataset
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── labels/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   │
│   ├── attributes-label/              # XML annotations
│   │   └── annotations_attributes/
│   │       └── video_XXXX_attributes.xml
│   │
│   ├── first-phase-model/             # Pre-trained YOLO
│   │   └── weights/
│   │       └── best.pt
│   │
│   ├── phase-3-dataset/               # Pre-processed sequences (optional)
│   │   ├── X_train.npy
│   │   ├── X_val.npy
│   │   ├── X_test.npy
│   │   └── data_scaler.pkl
│   │
│   └── phase-3-lstm-yolo/             # Trained LSTM (optional)
│       └── Phase_3_models/
│           └── lstm_intention_model.pth
│
└── working/                           # Writable output directory
    ├── data.yaml                      # YOLO configuration
    ├── master_train_dataset.csv       # Extracted features
    ├── master_val_dataset.csv
    ├── master_test_dataset.csv
    ├── balanced_train_dataset.csv     # Balanced training data
    ├── X_train.npy                   # LSTM sequences
    ├── X_val.npy
    ├── X_test.npy
    ├── training_results/               # YOLO training outputs
    │   └── jaad_final_model/
    │       └── weights/
    │           └── best.pt
    ├── evaluation/                    # Evaluation results
    │   ├── confusion_matrix_*.png
    │   └── classification_report_*.txt
    ├── final_results/                 # Processed videos
    │   └── result_video_XXXX.mp4
    └── lstm_intention_model.pth       # Trained LSTM model
```

### Local Project Structure (if cloning)

```
pedestrian-intention-prediction/
│
├── notebooks/
│   └── Pedestrian-Intuition.ipynb     # Kaggle notebook (downloadable)
│
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── LICENSE                             # License file
```

---

## Technical Details

### CLAHE Enhancement

```python
def enhance_low_light(img):
    """Apply CLAHE in LAB color space for low-light enhancement"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
```

**Impact**: Doubles average detection streak length in dark scenes

### Feature Extraction

Features are computed frame-by-frame for each tracked pedestrian:

- **Position**: `cx = (x1+x2)/2`, `cy = (y1+y2)/2`
- **Velocity**: `vel_x = cx_t - cx_{t-1}`, `vel_y = cy_t - cy_{t-1}`
- **Scale Change**: `delta_area = (w*h)_t / (w*h)_{t-1}`
- **Aspect Ratio**: `aspect_ratio = w/h`

### Sequence Building

- **Window Size**: 10 frames
- **Stride**: 1 frame (overlapping windows)
- **Minimum Track Length**: 5 frames (filtered out shorter tracks)
- **Normalization**: StandardScaler (fit on train, transform on val/test)

### Model Training

- **Loss Function**: `BCEWithLogitsLoss` with `pos_weight` for class imbalance
- **Early Stopping**: Patience=15 epochs, monitor validation loss
- **Learning Rate**: 0.0005 with weight decay 1e-3
- **Batch Size**: 64 sequences per batch

---

## Results

### Training Curves

The model achieves convergence after ~4 epochs with:
- **Training Loss**: Decreases from 0.65 to 0.35
- **Validation Loss**: Decreases from 0.60 to 0.32
- **Training Accuracy**: Increases from 65% to 85%
- **Validation Accuracy**: Increases from 68% to 84%

### Confusion Matrix

```
                Predicted
              Stay  Cross
Actual Stay   60%   40%
      Cross   14%   86%
```

### Key Findings

1. **High Precision for Crossing**: 96% precision means very few false alarms
2. **Good Recall for Crossing**: 86% recall captures most dangerous scenarios
3. **Temporal Latency**: Some errors occur when pedestrian transitions faster than 10-frame window (~0.3-0.5 seconds)
4. **Safety Recommendation**: Use asymmetric threshold (trigger at 30% crossing probability, resume at 10%)

---

## Requirements

### Core Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
Pillow>=10.0.0
pyyaml>=6.0
joblib>=1.3.0
```

### Optional Dependencies

```
jupyter>=1.0.0          # For notebook execution
tensorboard>=2.13.0     # For training visualization
```

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB+ for dataset and models

---

## Future Work

- [x] YOLO + LSTM pipeline implementation
- [x] CLAHE enhancement for low-light scenes
- [x] Real-time video processing
- [ ] Integration with Vision Transformer (ViT) for improved feature extraction
- [ ] Multi-modal fusion (combining visual and kinematic features)
- [ ] Real-time deployment on edge devices (NVIDIA Jetson)
- [ ] Extension to multi-pedestrian scenarios
- [ ] Uncertainty quantification for safety-critical applications
- [ ] Integration with vehicle control systems

---


## Contributors

- **Civan Arda Ozel** - *Co-author* - [Kaggle Profile](https://www.kaggle.com/cardaozel)
- **Kimiakarbasi** - *Co-author* - [Kaggle Profile](https://www.kaggle.com/kimiakarbasi)
- **Andres Cabilon** - *Co-author* - [Kaggle Profile](https://www.kaggle.com/andressabillon)
  

---

## Disclaimer

This project is for research and educational purposes. The models are trained on specific datasets and may not generalize to all scenarios. For safety-critical applications, thorough testing and validation are required.

**If you find this project useful, please consider giving it a star!**

---
## Citation

If you use this project in your research, please cite:

```bibtex
@misc{pedestrian-intention-prediction,
  title={Pedestrian Intention Prediction for Autonomous Vehicles},
  author={Kimiakarbasi},
  year={2026},
  howpublished={\url{https://www.kaggle.com/code/kimiakarbasi/pedestrian-intuition-phase-2-again}}
}


@inproceedings{jaad2017,
  title={JAAD: A Joint Attention Dataset for Autonomous Driving},
  author={Rasouli, Amir and Kotseruba, Iuliia and Tsotsos, John K.},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops (ICCVW)},
  year={2017},
  pages={1--8}
}
```
---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

