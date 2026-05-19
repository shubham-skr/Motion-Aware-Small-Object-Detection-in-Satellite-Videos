# Motion-Aware Optical flow guided temporal attention for small object detection in Statellite Videos.

A PyTorch implementation of a temporal small object detector that combines ConvLSTM-based motion modeling, temporal self-attention, and optical flow-guided feature warping with a CenterNet detection head — all on top of a DLA-34 backbone.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)

---

## Overview

We have given 5 files. The file names are self explanatory.

## Notebooks Included

- **micpl-bifpn-final-code.ipynb** — Implements the MICPL model with BiFPN for enhanced multi-scale feature fusion.
- **micpl-combined-final-with-2-only-no-bifpn.ipynb** — Combined MICPL architecture without BiFPN for a simplified and efficient pipeline.
- **micpl-combined.ipynb** — Base combined model integrating all components before final refinements.
- **micpl-final-raft.ipynb** — MICPL model integrated with RAFT for optical flow-based feature enhancement.
- **micpl-temporal-self-attention-final.ipynb** — Incorporates temporal self-attention to capture sequential dependencies in data.

The dataset is also given in VISO.

## Dataset

The project uses the **VISO** dataset in COCO format, structured for **aerial/plane detection**.

### Expected Directory Layout

```
VISO/
└── coco/
    └── plane/
        ├── train2017/          # training frame images
        ├── val2017/            # validation frame images
        ├── test2017/           # test frame images
        └── Annotations/
            ├── instances_train2017.json
            ├── instances_val2017.json
            └── instances_test2017.json
```

---

## Requirements

```
torch
torchvision          # >= 0.13 for RAFT-small
timm                 # DLA-34 from HuggingFace Hub
albumentations
pycocotools
opencv-python
numpy
matplotlib
tqdm
```

Install dependencies:

```bash
pip install torch torchvision timm albumentations pycocotools opencv-python numpy matplotlib tqdm
```

> **Note:** `torchvision.models.optical_flow.raft_small` requires `torchvision >= 0.13`.

---

## Quick Start

### 1. Prepare the Dataset

Download the VISO dataset and place it under `./VISO/coco/plane/` following the directory layout described above.

### 2. Run Training

Open the notebook and execute all cells in order, or run the training block directly. The training loop handles:

- Dataset and DataLoader instantiation
- Model and head construction
- Optimizer and scheduler setup
- Epoch-by-epoch training, validation, and checkpointing

### 3. Evaluate on the Test Set

```python
# Load best checkpoint
checkpoint = torch.load("./saved_models/best_micpl_model.pth", map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
head.load_state_dict(checkpoint['head_state_dict'])

# Run evaluation
f1, precision, recall = evaluate(model, head, test_loader, DEVICE)
```

### 4. Visualize Predictions

```python
visualize_sample(model, head, val_loader, DEVICE)
```

---

## Configuration

All major hyperparameters are defined in the `__main__` block:

```python
NUM_EPOCHS    = 20
BATCH_SIZE    = 4 * max(NUM_GPUS, 1)
LEARNING_RATE = 1.25e-4
CHANNELS      = 64
SEQ_LEN       = 5
NUM_CLASSES   = 1
```
