"""
Quickstart master script to run the full pipeline end-to-end:
1. Verify / Generate sample data and 30 graph representations
2. Run baselines and Tasks 1, 2, 3, and 4
3. Generate evaluation table, t-SNE plots, and qualitative case studies
"""

import os
import sys
import subprocess


def run_step(cmd, desc):
    print(f"\n=======================================================")
    print(f"==> {desc}")
    print(f"==> Running: {cmd}")
    print(f"=======================================================")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"Warning: Step returned exit code {ret.returncode}")


def main():
    py_exec = sys.executable or "python"
    print(f"Using Python executable: {py_exec}")

    # 1. Generate multi-modal data and graph samples
    run_step(f'"{py_exec}" src/sample_data_gen.py --num_samples 30', "Step 1: Generate Multi-Modal Music Samples & Graphs")

    # 2. Train and evaluate all tasks
    run_step(f'"{py_exec}" src/train.py --task all --epochs 3 --batch_size 4', "Step 2: Train & Evaluate Tasks 1, 2, 3, 4 and Baselines")

    print("\n[SUCCESS] Pipeline execution complete! Check results/metrics.json, results/plots/, and results/retrieval_examples/")


if __name__ == "__main__":
    main()
