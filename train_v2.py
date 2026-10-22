"""
TruthLens AI V2 — Master Training and Model Selection Pipeline
==============================================================
Trains V2 text, image, URL, consistency, and adaptive fusion modules.
Strictly adheres to:
  - NO TEST DATA USED DURING TRAINING OR TUNING
  - All hyperparameter tuning, model selection, calibration, and threshold optimization
    are conducted strictly on TRAIN and VALIDATION splits.
  - Frozen 600-sample test set remains untouched until evaluate_v2.py.
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.adaptive_fusion.train_fusion import load_cached_features
from modules.v2.text_encoder import TruthLensTextEncoderV2
from modules.v2.image_encoder import TruthLensImageEncoderV2
from modules.v2.url_analyzer import URLIntelligenceV2
from modules.v2.adaptive_fusion_v2 import TruthLensClassifierV2, V2_MODELS_DIR

EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiments", "exp_008_full_v2")
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
REPORTS_V2_DIR = os.path.join(PROJECT_ROOT, "reports", "v2")
os.makedirs(REPORTS_V2_DIR, exist_ok=True)


def run_train_v2():
    t_start = datetime.datetime.now()
    print("=" * 65)
    print("TRUTHLENS AI V2 — SAFE TRAINING & VALIDATION PIPELINE")
    print("=" * 65)

    # 1. Load data for Train and Validation splits ONLY
    print("\n[STEP 1] Loading Train and Validation datasets...")
    train_t, train_i, train_df = load_cached_features("train")
    val_t, val_i, val_df = load_cached_features("validation")

    y_train = train_df["label"].values.astype(int)
    y_val = val_df["label"].values.astype(int)

    print(f"Loaded Train: {len(train_df)} samples (REAL={int(np.sum(y_train==0))}, FAKE={int(np.sum(y_train==1))})")
    print(f"Loaded Validation: {len(val_df)} samples (REAL={int(np.sum(y_val==0))}, FAKE={int(np.sum(y_val==1))})")

    # 2. Train V2 Text Encoder Head on RoBERTa Embeddings
    print("\n[STEP 2] Training V2 Text Encoder Head on RoBERTa Embeddings...")
    X_text_tr = np.array([train_t[str(s)] for s in train_df["sample_id"]], dtype=np.float32)
    X_text_val = np.array([val_t[str(s)] for s in val_df["sample_id"]], dtype=np.float32)

    text_encoder = TruthLensTextEncoderV2()
    text_meta = text_encoder.fit(X_text_tr, y_train, X_text_val, y_val)

    # Generate text probabilities (P(fake))
    p_text_tr = text_encoder.classifier_head.predict_proba(X_text_tr)[:, 1]
    p_text_val = text_encoder.classifier_head.predict_proba(X_text_val)[:, 1]

    # 3. Train V2 Image Encoder Head on ViT Embeddings
    print("\n[STEP 3] Training V2 Image Encoder Head on ViT Embeddings...")
    tr_img_sids = [str(s) for s in train_df["sample_id"] if str(s) in train_i]
    X_img_tr_valid = np.array([train_i[s] for s in tr_img_sids], dtype=np.float32)
    y_img_tr_valid = train_df.set_index(train_df["sample_id"].astype(str)).loc[tr_img_sids, "label"].values.astype(int)

    val_img_sids = [str(s) for s in val_df["sample_id"] if str(s) in val_i]
    X_img_val_valid = np.array([val_i[s] for s in val_img_sids], dtype=np.float32)
    y_img_val_valid = val_df.set_index(val_df["sample_id"].astype(str)).loc[val_img_sids, "label"].values.astype(int)

    image_encoder = TruthLensImageEncoderV2()
    img_meta = image_encoder.fit(X_img_tr_valid, y_img_tr_valid, X_img_val_valid, y_img_val_valid)

    # Generate image probabilities (0.5 for missing)
    def get_img_prob(sid, idict, clf):
        if str(sid) in idict:
            arr = np.array(idict[str(sid)], dtype=np.float32).reshape(1, -1)
            return float(clf.predict_proba(arr)[0, 1])
        return 0.50

    p_img_tr = np.array([get_img_prob(s, train_i, image_encoder.classifier_head) for s in train_df["sample_id"]])
    p_img_val = np.array([get_img_prob(s, val_i, image_encoder.classifier_head) for s in val_df["sample_id"]])

    # 4. Train URL Intelligence on Train Split
    print("\n[STEP 4] Fitting URL Intelligence on Train Split...")
    url_analyzer = URLIntelligenceV2()
    p_url_tr = np.array([1.0 - url_analyzer.analyze(u)["url_reliability"] if url_analyzer.analyze(u)["url_reliability"] is not None else 0.50 for u in train_df["image_url"]])
    p_url_val = np.array([1.0 - url_analyzer.analyze(u)["url_reliability"] if url_analyzer.analyze(u)["url_reliability"] is not None else 0.50 for u in val_df["image_url"]])

    # 5. Build Availability Masks
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

    # 6. Fit and Select Multimodal Fusion Architecture
    print("\n[STEP 5] Candidate Architecture Selection, Threshold Tuning & Calibration...")
    fusion_clf = TruthLensClassifierV2()
    selection_meta = fusion_clf.fit_and_select(
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

    t_end = datetime.datetime.now()
    duration = (t_end - t_start).total_seconds()

    # 7. Record Experiment and Reproducibility
    repro_data = {
        "experiment_id": "exp_008_full_v2",
        "timestamp": t_end.strftime("%Y-%m-%d %H:%M:%S"),
        "training_duration_seconds": round(duration, 2),
        "random_seed": 42,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples_frozen": 600,
        "text_model_meta": text_meta,
        "image_model_meta": img_meta,
        "fusion_selection_meta": selection_meta,
        "library_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgb.__version__,
            "joblib": joblib.__version__
        }
    }

    with open(os.path.join(REPORTS_V2_DIR, "reproducibility.json"), "w", encoding="utf-8") as f:
        json.dump(repro_data, f, indent=2)

    with open(os.path.join(EXPERIMENT_DIR, "experiment_summary.json"), "w", encoding="utf-8") as f:
        json.dump(repro_data, f, indent=2)

    print("\n" + "=" * 65)
    print("V2 TRAINING & SELECTION COMPLETED SUCCESSFULLY!")
    print(f"Duration: {duration:.2f}s")
    print(f"Winning Architecture: {selection_meta['selected_architecture']}")
    print(f"Optimal Threshold: {selection_meta['optimal_threshold']}")
    print(f"Validation Accuracy: {selection_meta['validation_accuracy']*100:.2f}% | Val F1: {selection_meta['validation_f1']*100:.2f}%")
    print(f"Brier (val): {selection_meta['calibration_brier_before']:.4f} -> {selection_meta['calibration_brier_after']:.4f}")
    print(f"ECE (val):   {selection_meta['calibration_ece_before']:.4f} -> {selection_meta['calibration_ece_after']:.4f}")
    print("=" * 65)

    return repro_data

if __name__ == "__main__":
    run_train_v2()
