"""Run the full TruthLens V2 training and evaluation pipeline.

This script orchestrates the sequential execution of all required steps:
1. **Generate journalistic embeddings** (if not already present).
2. **Retrain the text classification head** using `retrain_text_head.py`.
3. **Retrain the image head** using `train_image_model.py`.
4. **Retrain the multimodal fusion model** using `train_fusion_model.py`.
5. **Run the final evaluation** on the frozen test set via `evaluate_v2.py`.

Each step is executed as a subprocess to preserve the original CLI behaviour and
argument handling of the individual scripts. Errors in any step abort the
pipeline, ensuring you can inspect logs before proceeding.
"""

import subprocess
import sys
import os
from pathlib import Path

# Helper to run a script and raise on failure
def run_script(script_path: str, description: str) -> None:
    script = Path(__file__).parent / script_path
    if not script.is_file():
        raise FileNotFoundError(f"{description} script not found: {script}")
    print(f"\n--- Running: {description} ({script.name}) ---")
    subprocess.run([sys.executable, str(script)], check=True)

def main() -> None:
    # Step 0: Ensure we are running from the project root
    os.chdir(Path(__file__).parent)

    # Step 1: Generate journalistic embeddings (optional – script handles caching)
    try:
        run_script("generate_journalistic_embeddings.py", "Journalistic embeddings generation")
    except Exception as e:
        print(f"[WARN] Skipping journalistic embeddings generation: {e}")

    # Step 2: Retrain text head
    run_script("retrain_text_head.py", "Text head retraining")

    # Step 3: Retrain image head
    run_script("train_image_model.py", "Image head retraining")

    # Step 4: Retrain fusion model
    run_script("train_fusion_model.py", "Fusion model retraining")

    # Step 5: Final evaluation
    run_script("evaluate_v2.py", "Final evaluation")

    print("\n[OK] Full pipeline completed successfully!")

if __name__ == "__main__":
    main()
