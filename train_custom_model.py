"""
Custom Security Model Training
================================
Fine-tunes YOLOv8 on neighborhood-specific security footage.
Supports: loitering, brawl, abandoned objects, fire/smoke, crowd surge, vehicle intrusion.

Requirements:
  pip install ultralytics supervision roboflow torch torchvision
"""

import os
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime


# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATASET_YAML   = "dataset/security_dataset.yaml"   # your annotated dataset
BASE_MODEL     = "yolov8n.pt"                       # n=nano, s=small, m=medium, l=large
PROJECT_NAME   = "security-model"
RUN_NAME       = f"run_{datetime.now().strftime('%Y%m%d_%H%M')}"
EPOCHS         = 100
IMG_SIZE       = 640
BATCH_SIZE     = 16
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
WORKERS        = 4
PATIENCE       = 20                                 # early stopping patience


# ─── DATASET YAML GENERATOR ───────────────────────────────────────────────────

def create_dataset_yaml(
    train_path: str,
    val_path: str,
    test_path: str,
    output_path: str = DATASET_YAML
):
    """
    Generate a dataset YAML for YOLOv8.
    Classes map to the security threat categories.
    """
    classes = {
        0: "person",
        1: "loitering",
        2: "brawl",
        3: "abandoned_bag",
        4: "fire",
        5: "smoke",
        6: "crowd_surge",
        7: "weapon",
        8: "vehicle_intrusion",
        9: "graffiti_vandalism",
        10: "suspicious_package",
    }

    data = {
        "train": train_path,
        "val":   val_path,
        "test":  test_path,
        "nc":    len(classes),
        "names": classes,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Dataset YAML written to: {output_path}")
    print(f"Classes: {classes}")
    return output_path


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train(
    dataset_yaml: str = DATASET_YAML,
    base_model:   str = BASE_MODEL,
    epochs:       int = EPOCHS,
    img_size:     int = IMG_SIZE,
    batch_size:   int = BATCH_SIZE,
    device:       str = DEVICE,
):
    """
    Fine-tune YOLOv8 on the security dataset.

    Augmentations applied automatically by Ultralytics:
      - Mosaic, MixUp, HSV shifts, flipping, rotation, perspective
      - Especially useful for rare events like brawls and fire
    """
    print(f"\n{'='*60}")
    print(f"  Training Security Model")
    print(f"  Base: {base_model}  |  Device: {device}  |  Epochs: {epochs}")
    print(f"{'='*60}\n")

    model = YOLO(base_model)

    results = model.train(
        data      = dataset_yaml,
        epochs    = epochs,
        imgsz     = img_size,
        batch     = batch_size,
        device    = device,
        workers   = WORKERS,
        patience  = PATIENCE,
        project   = PROJECT_NAME,
        name      = RUN_NAME,
        # Augmentation overrides
        mosaic    = 1.0,
        mixup     = 0.15,
        flipud    = 0.3,
        degrees   = 10,
        # Logging
        verbose   = True,
        save      = True,
        save_period = 10,        # checkpoint every N epochs
    )

    best_model_path = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"
    print(f"\nBest model saved to: {best_model_path}")
    print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
    print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
    return str(best_model_path)


# ─── EXPORT ───────────────────────────────────────────────────────────────────

def export_for_edge(model_path: str, format: str = "onnx"):
    """
    Export trained model for edge deployment.
    Supported formats: onnx, tensorrt, openvino, coreml, tflite
    """
    model = YOLO(model_path)
    export_path = model.export(format=format, half=True, simplify=True)
    print(f"Edge model exported ({format.upper()}): {export_path}")
    return export_path


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Generate dataset config (edit paths to your actual dataset)
    create_dataset_yaml(
        train_path = "dataset/images/train",
        val_path   = "dataset/images/val",
        test_path  = "dataset/images/test",
    )

    # 2. Fine-tune the model
    best_weights = train()

    # 3. Export for deployment on edge device (NVIDIA Jetson → TensorRT)
    export_for_edge(best_weights, format="onnx")
    # Uncomment for TensorRT (requires NVIDIA GPU):
    # export_for_edge(best_weights, format="engine")
