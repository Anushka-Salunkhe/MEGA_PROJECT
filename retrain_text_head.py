"""
TruthLens AI — Improved Text Classifier Head Retraining
=======================================================
Retrains the RoBERTa classification head using:
  1. Original multimodal train/validation splits (raev_train/val_text_features.json)
  2. Journalistic news dataset (PolitiFact + GossipCop titles — journalistic_train_text_features.json)

The combined dataset gives the head exposure to:
  - Short GossipCop/Reddit captions (original multimodal)
  - Full journalistic news headlines from PolitiFact + GossipCop (new)

This dramatically improves REAL recall — preventing real news from being misclassified as FAKE.
"""

import os
import sys
import json
import time
import numpy as np
import joblib

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "v2", "text_best")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES_DIR = os.path.join(PROJECT_ROOT, "storage", "features")


def load_json_embeddings(path):
    """Load embeddings from JSON file — handles both 'embedding' and direct list."""
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    X, y = [], []
    for item in items:
        emb = item.get("embedding") or item.get("emb")
        lbl = item.get("label", item.get("y", 0))
        if emb is not None:
            X.append(emb)
            y.append(int(lbl))
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


def run_retrain_text_head():
    print("=" * 65)
    print("TRUTHLENS AI — IMPROVED TEXT HEAD RETRAINING")
    print("=" * 65)
    t_start = time.time()

    # ----------------------------------------------------------------
    # 1. Load Original Multimodal Splits (train.csv embeddings)
    # ----------------------------------------------------------------
    print("\n[STEP 1] Loading original multimodal embeddings...")
    train_file = os.path.join(FEATURES_DIR, "raev_train_text_features.json")
    val_file = os.path.join(FEATURES_DIR, "raev_validation_text_features.json")

    X_train_orig, y_train_orig = load_json_embeddings(train_file)
    X_val, y_val = load_json_embeddings(val_file)

    print(f"  Original Train: {X_train_orig.shape} | REAL={np.sum(y_train_orig==0)}, FAKE={np.sum(y_train_orig==1)}")
    print(f"  Validation:     {X_val.shape} | REAL={np.sum(y_val==0)}, FAKE={np.sum(y_val==1)}")

    # ----------------------------------------------------------------
    # 2. Load Journalistic News Embeddings (PolitiFact + GossipCop)
    # ----------------------------------------------------------------
    print("\n[STEP 2] Loading journalistic news embeddings...")
    journ_file = os.path.join(FEATURES_DIR, "journalistic_train_text_features.json")

    if not os.path.exists(journ_file):
        print(f"  [WARNING] {journ_file} not found — skipping journalistic augmentation")
        X_combined = X_train_orig
        y_combined = y_train_orig
    else:
        X_journ, y_journ = load_json_embeddings(journ_file)
        print(f"  Journalistic: {X_journ.shape} | REAL={np.sum(y_journ==0)}, FAKE={np.sum(y_journ==1)}")

        # ----------------------------------------------------------------
        # 3. Combine Training Data
        # ----------------------------------------------------------------
        X_combined = np.concatenate([X_train_orig, X_journ], axis=0)
        y_combined = np.concatenate([y_train_orig, y_journ], axis=0)

    print(f"\n[STEP 3] Combined Training Set: {X_combined.shape}")
    print(f"  REAL={np.sum(y_combined==0)}, FAKE={np.sum(y_combined==1)}")

    # ----------------------------------------------------------------
    # 4. Train Improved Classification Head
    # ----------------------------------------------------------------
    print("\n[STEP 4] Training Improved Logistic Regression Head...")
    # Higher C = less regularization (allows more nuance)
    # class_weight='balanced' ensures REAL and FAKE are equally weighted
    head = LogisticRegression(
        C=3.0,
        max_iter=3000,
        random_state=42,
        class_weight='balanced',
        solver='lbfgs'
    )
    head.fit(X_combined, y_combined)

    # ----------------------------------------------------------------
    # 5. Evaluate on Validation Set
    # ----------------------------------------------------------------
    print("\n[STEP 5] Evaluating on Validation Set...")
    val_probs = head.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    val_acc = float(accuracy_score(y_val, val_preds))
    val_f1 = float(f1_score(y_val, val_preds, zero_division=0))
    val_auc = float(roc_auc_score(y_val, val_probs))

    # Also check REAL recall specifically
    real_mask = y_val == 0
    fake_mask = y_val == 1
    real_recall = float(np.mean(val_preds[real_mask] == 0))
    fake_recall = float(np.mean(val_preds[fake_mask] == 1))

    print(f"  Validation Accuracy:  {val_acc*100:.2f}%")
    print(f"  Validation F1:        {val_f1*100:.2f}%")
    print(f"  Validation ROC-AUC:   {val_auc:.4f}")
    print(f"  REAL Recall:          {real_recall*100:.2f}%")
    print(f"  FAKE Recall:          {fake_recall*100:.2f}%")

    # ----------------------------------------------------------------
    # 6. Save Model
    # ----------------------------------------------------------------
    head_path = os.path.join(MODEL_DIR, "text_classifier_head.pkl")
    meta_path = os.path.join(MODEL_DIR, "text_encoder_meta.json")

    joblib.dump(head, head_path)

    meta = {
        "model_type": "RoBERTa-base + Improved Task Head (Journalistic + Multimodal)",
        "embedding_dim": 768,
        "validation_accuracy": round(val_acc, 4),
        "validation_f1": round(val_f1, 4),
        "validation_auc": round(val_auc, 4),
        "validation_real_recall": round(real_recall, 4),
        "validation_fake_recall": round(fake_recall, 4),
        "regularization_C": 3.0,
        "class_weight": "balanced",
        "training_data": "multimodal_train + politifact + gossipcop_titles",
        "combined_train_samples": len(y_combined)
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    duration = time.time() - t_start
    print("\n" + "=" * 65)
    print("TEXT HEAD RETRAINING COMPLETED!")
    print(f"Duration:         {duration:.2f}s")
    print(f"Validation Acc:   {val_acc*100:.2f}%")
    print(f"Validation F1:    {val_f1*100:.2f}%")
    print(f"Validation AUC:   {val_auc:.4f}")
    print(f"REAL Recall:      {real_recall*100:.2f}%")
    print(f"FAKE Recall:      {fake_recall*100:.2f}%")
    print(f"Model saved to:   {head_path}")
    print("=" * 65)

    return meta


if __name__ == "__main__":
    run_retrain_text_head()
