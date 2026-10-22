"""
TruthLens AI — Image Model (Vision Transformer) Training & Fine-Tuning Pipeline
================================================================================
Trains and calibrates the visual forensic head on ViT-base-patch16-224 (768-dim) embeddings.
Uses strictly TRAIN and VALIDATION splits — frozen test set is preserved untouched.

Outputs:
  - models/v2/image_best/classifier_head.joblib
  - models/v2/image_best/metadata.json
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.adaptive_fusion.train_fusion import load_cached_features
from modules.v2.image_encoder import TruthLensImageEncoderV2


def run_train_image_model():
    print("=" * 65)
    print("TRUTHLENS AI — VISION TRANSFORMER (ViT) HEAD TRAINING")
    print("=" * 65)
    t_start = time.time()

    print("\n[STEP 1] Loading Train and Validation image embeddings...")
    _, train_i, train_df = load_cached_features("train")
    _, val_i, val_df = load_cached_features("validation")

    tr_img_sids = [str(s) for s in train_df["sample_id"] if str(s) in train_i]
    X_img_tr = np.array([train_i[s] for s in tr_img_sids], dtype=np.float32)
    y_img_tr = train_df.set_index(train_df["sample_id"].astype(str)).loc[tr_img_sids, "label"].values.astype(int)

    val_img_sids = [str(s) for s in val_df["sample_id"] if str(s) in val_i]
    X_img_val = np.array([val_i[s] for s in val_img_sids], dtype=np.float32)
    y_img_val = val_df.set_index(val_df["sample_id"].astype(str)).loc[val_img_sids, "label"].values.astype(int)

    print(f"Loaded Train Images: {len(X_img_tr)} samples (Shape: {X_img_tr.shape})")
    print(f"Loaded Val Images:   {len(X_img_val)} samples (Shape: {X_img_val.shape})")
    print(f"Train Image Class Balance: REAL = {np.sum(y_img_tr == 0)}, FAKE = {np.sum(y_img_tr == 1)}")
    print(f"Val Image Class Balance:   REAL = {np.sum(y_img_val == 0)}, FAKE = {np.sum(y_img_val == 1)}")

    print("\n[STEP 2] Fitting ViT Classification Head (Balanced Logistic + Platt Calibrator)...")
    image_encoder = TruthLensImageEncoderV2()
    meta = image_encoder.fit(X_img_tr, y_img_tr, X_img_val, y_img_val)

    duration = time.time() - t_start
    print("\n" + "=" * 65)
    print("IMAGE MODEL TRAINING COMPLETED!")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Validation Accuracy: {meta.get('validation_accuracy', 0.0)*100:.2f}%")
    print(f"Validation F1:       {meta.get('validation_f1', 0.0)*100:.2f}%")
    print(f"Validation ROC-AUC:  {meta.get('validation_auc', 0.0):.4f}")
    print(f"Checkpoints saved to: {image_encoder.model_dir}")
    print("=" * 65)

    return meta


if __name__ == "__main__":
    run_train_image_model()
