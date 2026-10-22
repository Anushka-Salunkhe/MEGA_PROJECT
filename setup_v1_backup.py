"""
TruthLens AI — Setup Directory Structure and V1 Backup (Local to working project)
"""

import os
import shutil
import json

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

DIRS_TO_CREATE = [
    os.path.join(PROJECT_ROOT, "models", "v1"),
    os.path.join(PROJECT_ROOT, "models", "v2"),
    os.path.join(PROJECT_ROOT, "outputs", "v1"),
    os.path.join(PROJECT_ROOT, "outputs", "v2"),
    os.path.join(PROJECT_ROOT, "evaluation", "v1"),
    os.path.join(PROJECT_ROOT, "evaluation", "v2"),
    os.path.join(PROJECT_ROOT, "reports", "v1"),
    os.path.join(PROJECT_ROOT, "reports", "v2"),
    os.path.join(PROJECT_ROOT, "experiments"),
    os.path.join(PROJECT_ROOT, "backup_v1")
]

for d in DIRS_TO_CREATE:
    os.makedirs(d, exist_ok=True)
    print(f"[CREATED] {d}")

# 1. Backup V1 outputs to outputs/v1 and backup_v1/outputs
outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
out_v1_dir = os.path.join(PROJECT_ROOT, "outputs", "v1")
if os.path.exists(outputs_dir):
    for fname in os.listdir(outputs_dir):
        fpath = os.path.join(outputs_dir, fname)
        if os.path.isfile(fpath):
            shutil.copy2(fpath, os.path.join(out_v1_dir, fname))
            print(f"[COPIED] {fname} -> outputs/v1/")

# 2. Backup V1 models to models/v1
storage_models_dir = os.path.join(PROJECT_ROOT, "storage", "models")
models_v1_dir = os.path.join(PROJECT_ROOT, "models", "v1")
if os.path.exists(storage_models_dir):
    for fname in os.listdir(storage_models_dir):
        fpath = os.path.join(storage_models_dir, fname)
        if os.path.isfile(fpath):
            shutil.copy2(fpath, os.path.join(models_v1_dir, fname))
            print(f"[COPIED] {fname} -> models/v1/")

# 3. Backup V1 evaluation script to evaluation/v1/
eval_py = os.path.join(PROJECT_ROOT, "evaluate.py")
eval_v1_py = os.path.join(PROJECT_ROOT, "evaluation", "v1", "evaluate_v1.py")
if os.path.exists(eval_py):
    shutil.copy2(eval_py, eval_v1_py)
    print(f"[COPIED] evaluate.py -> evaluation/v1/evaluate_v1.py")

benchmark_py = os.path.join(PROJECT_ROOT, "modules", "evaluation", "benchmark.py")
if os.path.exists(benchmark_py):
    shutil.copy2(benchmark_py, os.path.join(PROJECT_ROOT, "evaluation", "v1", "benchmark_v1.py"))

# 4. Copy current evaluation results to reports/v1/
eval_json = os.path.join(outputs_dir, "evaluation_results.json")
if os.path.exists(eval_json):
    shutil.copy2(eval_json, os.path.join(PROJECT_ROOT, "reports", "v1", "evaluation_results_v1.json"))
eval_txt = os.path.join(outputs_dir, "evaluation_summary.txt")
if os.path.exists(eval_txt):
    shutil.copy2(eval_txt, os.path.join(PROJECT_ROOT, "reports", "v1", "evaluation_summary_v1.txt"))

# 5. Full backup of critical V1 source files into backup_v1/
backup_root = os.path.join(PROJECT_ROOT, "backup_v1")
files_to_backup = [
    "app.py", "evaluate.py", "test_text.py", "test_image.py",
    os.path.join("modules", "adaptive_fusion", "reliability_fusion.py"),
    os.path.join("modules", "adaptive_fusion", "xgboost_classifier.py"),
    os.path.join("modules", "adaptive_fusion", "feature_projector.py"),
    os.path.join("modules", "adaptive_fusion", "train_fusion.py"),
    os.path.join("modules", "raev", "raev_fusion.py"),
    os.path.join("modules", "raev", "raev_evaluator.py"),
    os.path.join("modules", "explainability", "shap_explainer.py"),
    os.path.join("controllers", "pipeline.py")
]

for rel_path in files_to_backup:
    src = os.path.join(PROJECT_ROOT, rel_path)
    dst = os.path.join(backup_root, rel_path)
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[BACKUP] {rel_path} -> backup_v1/{rel_path}")

print("\n[SUCCESS] Local V1 Backup complete!")
