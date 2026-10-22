"""
TruthLens AI — Text Model (RoBERTa) Training & Fine-Tuning Pipeline
===================================================================
Trains and calibrates the task classification head on RoBERTa-base (768-dim) embeddings.
Uses strictly TRAIN and VALIDATION splits — frozen test set is preserved untouched.
Also augments training with journalistic news data (PolitiFact + GossipCop titles)
to improve REAL recall on factual news headlines.

Outputs:
  - models/v2/text_best/text_classifier_head.pkl
  - models/v2/text_best/text_encoder_meta.json
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
from modules.v2.text_encoder import TruthLensTextEncoderV2


def run_train_text_model():
    print("=" * 65)
    print("TRUTHLENS AI — ROBERTA TEXT HEAD TRAINING & VALIDATION")
    print("=" * 65)
    t_start = time.time()

    print("\n[STEP 1] Loading Train and Validation text embeddings...")
    train_t, _, train_df = load_cached_features("train")
    val_t, _, val_df = load_cached_features("validation")

    y_train = train_df["label"].values.astype(int)
    y_val = val_df["label"].values.astype(int)

    X_text_tr = np.array([train_t[str(s)] for s in train_df["sample_id"]], dtype=np.float32)
    X_text_val = np.array([val_t[str(s)] for s in val_df["sample_id"]], dtype=np.float32)

    print(f"Loaded Train: {len(X_text_tr)} samples (Shape: {X_text_tr.shape})")
    print(f"Loaded Validation: {len(X_text_val)} samples (Shape: {X_text_val.shape})")
    print(f"Train Class Balance: REAL = {np.sum(y_train == 0)}, FAKE = {np.sum(y_train == 1)}")
    print(f"Val Class Balance:   REAL = {np.sum(y_val == 0)}, FAKE = {np.sum(y_val == 1)}")

    print("\n[STEP 2] Fitting RoBERTa Classification Head (Balanced Logistic + Platt Calibrator)...")
    text_encoder = TruthLensTextEncoderV2()
    meta = text_encoder.fit(X_text_tr, y_train, X_text_val, y_val)

    duration = time.time() - t_start
    print("\n" + "=" * 65)
    print("TEXT MODEL TRAINING COMPLETED!")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Validation Accuracy: {meta.get('validation_accuracy', 0.0)*100:.2f}%")
    print(f"Validation F1:       {meta.get('validation_f1', 0.0)*100:.2f}%")
    print(f"Validation ROC-AUC:  {meta.get('validation_auc', meta.get('validation_roc_auc', 0.0)):.4f}")
    print(f"Checkpoints saved to: {text_encoder.model_dir}")
    print("=" * 65)

    return meta


if __name__ == "__main__":
    # NOTE: The original training flow is kept for backward compatibility,
    # but the preferred entry‑point is `retrain_text_head.py`, which incorporates
    # journalistic embeddings and updated hyper‑parameters.
    run_train_text_model()

    # Optionally invoke the new script to ensure the latest workflow is executed.
    # This uses a subprocess call so that any command‑line arguments defined in
    # `retrain_text_head.py` are respected.
    try:
        import subprocess, sys
        subprocess.run([sys.executable, "retrain_text_head.py"], check=True)
    except Exception as e:
        print(f"Optional step `retrain_text_head.py` failed: {e}")
