"""
TruthLens AI — Multimodal Adaptive Fusion & XGBoost Training Pipeline
======================================================================
Trains and evaluates candidate fusion architectures strictly on TRAIN and VALIDATION splits.
Performs threshold optimization and Platt probability calibration.
Preserves the frozen 600-sample test set untouched.

Outputs:
  - models/v2/v2_classifier_meta.json
  - models/v2/v2_classifier.joblib
  - models/v2/v2_calibrator.joblib
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
from modules.v2.image_encoder import TruthLensImageEncoderV2
from modules.v2.url_analyzer import URLIntelligenceV2
from modules.v2.adaptive_fusion_v2 import TruthLensClassifierV2, V2_MODELS_DIR


def run_train_fusion_model():
    print("=" * 65)
    print("TRUTHLENS AI — MULTIMODAL ADAPTIVE FUSION TRAINING")
    print("=" * 65)
    t_start = time.time()

    # 1. Load Train and Validation Splits
    print("\n[STEP 1] Loading Train & Validation splits...")
    train_t, train_i, train_df = load_cached_features("train")
    val_t, val_i, val_df = load_cached_features("validation")

    y_train = train_df["label"].values.astype(int)
    y_val = val_df["label"].values.astype(int)

    # 2. Extract Modality Probabilities (using trained encoders)
    print("\n[STEP 2] Extracting Out-of-Fold Modality Probabilities...")
    text_encoder = TruthLensTextEncoderV2()
    image_encoder = TruthLensImageEncoderV2()
    url_analyzer = URLIntelligenceV2()

    X_text_tr = np.array([train_t[str(s)] for s in train_df["sample_id"]], dtype=np.float32)
    X_text_val = np.array([val_t[str(s)] for s in val_df["sample_id"]], dtype=np.float32)

    p_text_tr = text_encoder.classifier_head.predict_proba(X_text_tr)[:, 1]
    p_text_val = text_encoder.classifier_head.predict_proba(X_text_val)[:, 1]

    def get_img_prob(sid, idict, clf):
        if str(sid) in idict:
            arr = np.array(idict[str(sid)], dtype=np.float32).reshape(1, -1)
            return float(clf.predict_proba(arr)[0, 1])
        return 0.50

    p_img_tr = np.array([get_img_prob(s, train_i, image_encoder.classifier_head) for s in train_df["sample_id"]])
    p_img_val = np.array([get_img_prob(s, val_i, image_encoder.classifier_head) for s in val_df["sample_id"]])

    p_url_tr = np.array([
        1.0 - url_analyzer.analyze(u)["url_reliability"]
        if url_analyzer.analyze(u)["url_reliability"] is not None else 0.50
        for u in train_df["image_url"]
    ])
    p_url_val = np.array([
        1.0 - url_analyzer.analyze(u)["url_reliability"]
        if url_analyzer.analyze(u)["url_reliability"] is not None else 0.50
        for u in val_df["image_url"]
    ])

    masks_tr = np.column_stack([
        np.ones(len(train_df)),
        [1.0 if str(s) in train_i else 0.0 for s in train_df["sample_id"]],
        [1.0 if str(u) and "http" in str(u) else 0.0 for u in train_df["image_url"]]
    ])
    masks_val = np.column_stack([
        np.ones(len(val_df)),
        [1.0 if str(s) in val_i else 0.0 for s in val_df["sample_id"]],
        [1.0 if str(u) and "http" in str(u) else 0.0 for u in val_df["image_url"]]
    ])

    # 3. Fit Candidate Architectures & Select Winner
    print("\n[STEP 3] Comparing Candidate Architectures & Platt Calibrating...")
    fusion_clf = TruthLensClassifierV2()
    meta = fusion_clf.fit_and_select(
        p_text_tr=p_text_tr,
        p_img_tr=p_img_tr,
        p_url_tr=p_url_tr,
        masks_tr=masks_tr,
        y_train=y_train,
        p_text_val=p_text_val,
        p_img_val=p_img_val,
        p_url_val=p_url_val,
        masks_val=masks_val,
        y_val=y_val
    )

    duration = time.time() - t_start
    print("\n" + "=" * 65)
    print("MULTIMODAL FUSION TRAINING COMPLETED!")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Selected Architecture: {meta['selected_architecture']}")
    print(f"Optimal Threshold:     {meta['optimal_threshold']}")
    print(f"Validation Accuracy:   {meta['validation_accuracy']*100:.2f}%")
    print(f"Validation F1:         {meta['validation_f1']*100:.2f}%")
    print(f"Validation ROC-AUC:    {meta['validation_roc_auc']:.4f}")
    print(f"Calibration Brier:     {meta['calibration_brier_before']:.4f} -> {meta['calibration_brier_after']:.4f}")
    print(f"Calibration ECE:       {meta['calibration_ece_before']:.4f} -> {meta['calibration_ece_after']:.4f}")
    print("=" * 65)

    return meta


if __name__ == "__main__":
    run_train_fusion_model()
