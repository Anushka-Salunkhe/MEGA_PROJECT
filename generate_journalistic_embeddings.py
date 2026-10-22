"""
TruthLens AI — Generate Journalistic News Embeddings
===================================================
Generates and caches RoBERTa-base contextual embeddings for balanced journalistic
news samples from PolitiFact and GossipCop datasets.

Used by `retrain_text_head.py` to augment the text classification head with
diverse journalistic writing patterns and improve REAL/FAKE discrimination.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUT_FILE = os.path.join(PROJECT_ROOT, "storage", "features", "journalistic_train_text_features.json")


def generate_embeddings(force: bool = False, max_samples: int = 3200):
    if os.path.exists(OUTPUT_FILE) and not force:
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if len(existing) >= max_samples:
                print(f"[OK] Journalistic embeddings already exist at {OUTPUT_FILE} ({len(existing)} samples). Use --force to regenerate.")
                return
        except Exception as e:
            print(f"[WARN] Error reading existing embeddings, regenerating: {e}")

    print("=" * 65)
    print("TRUTHLENS AI — GENERATE JOURNALISTIC EMBEDDINGS")
    print("=" * 65)

    from utils.text_processor import processor as base_text_processor

    datasets_dir = os.path.join(PROJECT_ROOT, "datasets")
    real_sources = [
        os.path.join(datasets_dir, "politifact_real.csv"),
        os.path.join(datasets_dir, "gossipcop_real.csv"),
    ]
    fake_sources = [
        os.path.join(datasets_dir, "politifact_fake.csv"),
        os.path.join(datasets_dir, "gossipcop_fake.csv"),
    ]

    def load_titles(files):
        titles = []
        for f in files:
            if os.path.exists(f):
                try:
                    df = pd.read_csv(f)
                    for col in ["title", "text", "statement"]:
                        if col in df.columns:
                            t = df[col].dropna().astype(str).str.strip()
                            t = t[t.str.len() > 15].tolist()
                            titles.extend(t)
                            break
                except Exception as ex:
                    print(f"  Warning reading {f}: {ex}")
        return list(dict.fromkeys(titles))  # remove duplicates while preserving order

    real_titles = load_titles(real_sources)
    fake_titles = load_titles(fake_sources)

    half = max_samples // 2
    np.random.seed(42)
    selected_real = np.random.choice(real_titles, size=min(half, len(real_titles)), replace=False)
    selected_fake = np.random.choice(fake_titles, size=min(half, len(fake_titles)), replace=False)

    print(f"Selected {len(selected_real)} REAL and {len(selected_fake)} FAKE journalistic samples.")

    entries = []
    print("Extracting RoBERTa embeddings...")
    for idx, t in enumerate(selected_real):
        emb = base_text_processor.roberta_embedding(t).flatten().tolist()
        entries.append({"text": str(t), "label": 0, "embedding": emb})
        if (idx + 1) % 400 == 0:
            print(f"  Processed {idx + 1}/{len(selected_real)} REAL samples")

    for idx, t in enumerate(selected_fake):
        emb = base_text_processor.roberta_embedding(t).flatten().tolist()
        entries.append({"text": str(t), "label": 1, "embedding": emb})
        if (idx + 1) % 400 == 0:
            print(f"  Processed {idx + 1}/{len(selected_fake)} FAKE samples")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f)

    print(f"[OK] Successfully saved {len(entries)} embeddings to {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate journalistic RoBERTa embeddings")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if file exists")
    parser.add_argument("--samples", type=int, default=3200, help="Total balanced samples to extract")
    args = parser.parse_args()
    generate_embeddings(force=args.force, max_samples=args.samples)
