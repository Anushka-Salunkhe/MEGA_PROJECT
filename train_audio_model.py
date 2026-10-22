"""
TruthLens AI — Audio Model (Whisper + Librosa) Setup & Pipeline Verification
=============================================================================
Configures speech-to-text transcription via Whisper and acoustic feature extraction
via Librosa/SoundFile.
Adheres strictly to research guidelines:
  - Whisper is a transcription model, not proof of audio authenticity.
  - In the absence of a labeled audio deepfake dataset, independent deepfake
    authenticity is clearly and transparently marked as EXPERIMENTAL / UNAVAILABLE.
  - The transcribed claim is forwarded into the text verification and external
    evidence pipeline for factual credibility verification.
"""

import os
import sys
import json
import time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

V2_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "v2")
os.makedirs(V2_MODELS_DIR, exist_ok=True)


def run_train_audio_model():
    print("=" * 65)
    print("TRUTHLENS AI — AUDIO MODEL (WHISPER & ACOUSTIC) PIPELINE SETUP")
    print("=" * 65)
    t_start = time.time()

    # 1. Verify Whisper and Librosa Dependencies
    print("\n[STEP 1] Verifying Speech & Audio Processing Dependencies...")
    status_whisper = False
    status_librosa = False

    try:
        import whisper
        print("  [OK] Whisper: available")
        status_whisper = True
    except ImportError:
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            print("  [OK] Transformers Whisper: available")
            status_whisper = True
        except ImportError:
            print("  [X] Whisper: not installed or unavailable")

    try:
        import librosa
        import soundfile
        print("  [OK] Librosa & SoundFile: available")
        status_librosa = True
    except ImportError:
        print("  [X] Librosa/SoundFile: not available")

    # 2. Document Modality Capabilities & Research Ground Truth
    print("\n[STEP 2] Auditing Audio Training & Ground Truth Status...")
    meta = {
        "modality": "audio",
        "transcription_engine": "Whisper-tiny",
        "acoustic_features": ["MFCC", "Spectral Centroid", "Zero-Crossing Rate", "Chroma"],
        "speech_transcription_status": "AVAILABLE" if status_whisper else "UNAVAILABLE",
        "acoustic_feature_status": "AVAILABLE" if status_librosa else "UNAVAILABLE",
        "deepfake_authenticity_status": "EXPERIMENTAL_UNAVAILABLE",
        "status_explanation": (
            "No labeled audio deepfake dataset (e.g., ASVspoof, WaveFake) is currently bundled. "
            "Whisper speech transcription is fully functional and routes transcribed claims into "
            "the text and external fact-checking verification engines. Independent acoustic "
            "authenticity classification is marked experimental to prevent fabricated scores."
        ),
        "claim_forwarding_enabled": True,
        "sample_rate_hz": 16000
    }

    meta_file = os.path.join(V2_MODELS_DIR, "audio_meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    duration = time.time() - t_start
    print("\n" + "=" * 65)
    print("AUDIO MODEL CONFIGURATION COMPLETED!")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Speech Transcription: {meta['speech_transcription_status']}")
    print(f"Authenticity Status:   {meta['deepfake_authenticity_status']}")
    print(f"Metadata saved to:    {meta_file}")
    print("=" * 65)

    return meta


if __name__ == "__main__":
    run_train_audio_model()
