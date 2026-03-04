from ultralytics import YOLO
import csv
import os
import shutil
import random
import numpy as np
import torch
from typing import List, Dict, Any

# --------------------
# Config
# --------------------
freeze_values = [0, 1, 2, 3, 5, 10, 15]
PROJECT_DIR = os.path.join("runs", "segment")
RUN_NAME_PREFIX = "tune_freeze"          # run dir will be tune_freeze_f{freeze}
SUMMARY_CSV = os.path.join("runs", "freeze_tuning_summary.csv")
CLEANUP_RUN_DIR = True                   # delete each run dir after reading results
SEED = 42

# --------------------
# Utils
# --------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def last_row(csv_path: str) -> Dict[str, Any]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")
    return rows[-1]

def ensure_summary_header(path: str, fieldnames: List[str]):
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    if not file_exists:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

def append_summary_row(path: str, fieldnames: List[str], row: Dict[str, Any]):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)
        f.flush()

# --------------------
# Run
# --------------------
set_global_seed(SEED)

# We don't know column names before first run; collect dynamically.
summary_fieldnames: List[str] = []

for freeze_value in freeze_values:
    # unique run name per freeze so results.csv is isolated
    run_name = f"{RUN_NAME_PREFIX}_f{freeze_value}"
    run_dir = os.path.join(PROJECT_DIR, run_name)

    model = YOLO("yolo11s-seg.pt")

    model.train(
        data="../data/aiforest_coco8.yaml",
        batch=0.7,
        epochs=80,
        device="0",
        name=run_name,
        overlap_mask=False,
        resume=False,
        dropout=0.5,
        multi_scale=True,
        optimizer="auto",
        cfg="best_hyperparameters.yaml",
        imgsz=320,
        freeze=freeze_value,
        nbs=64,
        seed=SEED,
        exist_ok=True,
        save=False,          # no final/best weights saved
        save_period=-1,      # no periodic checkpoints
        plots=False,
        deterministic=True,
        project=PROJECT_DIR, # ensure runs/segment/<run_name>
        fraction=0.5
    )

    results_csv = os.path.join(run_dir, "results.csv")
    if not os.path.exists(results_csv):
        print(f"Results file not found for freeze={freeze_value} at {results_csv}")
        continue

    row = last_row(results_csv)
    loss_cols = {k: v for k, v in row.items() if "loss" in k}
    metrics = {k: v for k, v in row.items() if "metrics/" in k or "mAP" in k}

    summary_row = {
        "freeze": freeze_value,
        "results_file": results_csv,
        **loss_cols,
        **metrics,
    }

    # Prepare summary header on first successful run
    if not summary_fieldnames:
        # freeze first, then sorted keys
        summary_fieldnames = ["freeze"] + sorted(
            [k for k in summary_row.keys() if k != "freeze"]
        )
        ensure_summary_header(SUMMARY_CSV, summary_fieldnames)

    # Append immediately after each training
    append_summary_row(SUMMARY_CSV, summary_fieldnames, summary_row)
    print(f"[freeze={freeze_value}] appended to {SUMMARY_CSV}")

    # Optional: clean up run directory to save space
    if CLEANUP_RUN_DIR:
        try:
            shutil.rmtree(run_dir)
            print(f"Deleted run directory: {run_dir}")
        except Exception as e:
            print(f"Warning: could not delete {run_dir}: {e}")

print(f"\nDone. Incremental summary available at: {SUMMARY_CSV}")
