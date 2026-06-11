#!/usr/bin/env python
# coding: utf-8
"""
scripts/00_train_vectors.py
===========================
Train CAA steering vectors for all layers of a model upfront.

Run this once before 01_layer_sweep.py. Vectors are saved to the
steer_vector_output_dirs specified in the config yaml.

Usage:
    cd ~/EasyEdit_clean
    python code/scripts/00_train_vectors.py
    python code/scripts/00_train_vectors.py --config code/configs/config_toxicity-gpt2.yaml
    python code/scripts/00_train_vectors.py --layers 2 6 10   # specific layers only
    python code/scripts/00_train_vectors.py --skip-existing   # skip already trained
"""

import os, sys, json, argparse
import torch
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR    = os.path.dirname(_SCRIPT_DIR)
_PROJECT_DIR = os.path.dirname(_CODE_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _CODE_DIR)

from utils import (
    load_config, load_config_raw, first_path,
    get_n_layers, set_generate_yaml_layer,
    find_vector_file, make_splits,
)
from steer.vector_generators.vector_generators import BaseVectorGenerator

# ============================================================================
# Defaults
# ============================================================================

DEFAULT_CONFIG   = os.path.join(_CODE_DIR, "configs", "config_toxicity-gpt2.yaml")
GENERATE_YAML    = os.path.join(_PROJECT_DIR, "hparams", "Steer", "caa_hparams", "generate_caa.yaml")
SPLIT_SEED        = 42
TRAIN_VECTOR_FRAC = 0.80


def train_layer(layer, train_data, config_path, dataset_key):
    """Train CAA vector for a single layer."""
    set_generate_yaml_layer(GENERATE_YAML, layer)
    cfg_raw   = load_config_raw(config_path)
    generator = BaseVectorGenerator(cfg_raw)
    generator.generate_vectors({dataset_key: train_data})


def main():
    parser = argparse.ArgumentParser(description="Train CAA vectors for all layers")
    parser.add_argument("--config",        default=DEFAULT_CONFIG,
                        help="Path to experiment config yaml")
    parser.add_argument("--layers",        nargs="+", type=int, default=None,
                        help="Specific layers to train (default: all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip layers that already have a saved vector")
    args = parser.parse_args()

    config_path = args.config
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg         = load_config(config_path)
    # Convert all OmegaConf values to plain Python types
    from omegaconf import OmegaConf
    cfg_plain   = OmegaConf.to_container(cfg, resolve=True)
    model_name  = str(cfg_plain.get("model_name_or_path", "gpt2"))
    vec_dir     = first_path(cfg_plain.get("steer_vector_load_dir"))
    dataset_key = cfg_plain.get("steer_train_dataset", ["toxicity"])
    if isinstance(dataset_key, list):
        dataset_key = str(dataset_key[0])
    else:
        dataset_key = str(dataset_key)
    n_layers    = get_n_layers(model_name)
    target_layers = args.layers if args.layers else list(range(n_layers))

    print("=" * 65)
    print("VECTOR TRAINING")
    print(f"Config:   {config_path}")
    print(f"Model:    {model_name}")
    print(f"Dataset:  {dataset_key}")
    print(f"Layers:   {target_layers}")
    print(f"Vec dir:  {vec_dir}")
    print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # Data split — only need train_data
    print(f"\n{'='*65}\nDATA SPLIT\n{'='*65}")
    train_data, _ = make_splits(
        cfg, dataset_key=dataset_key,
        train_frac=TRAIN_VECTOR_FRAC,
        seed=SPLIT_SEED,
    )

    # Train vectors
    results = {}
    for layer in target_layers:
        print(f"\n{'─'*65}\nLayer {layer}/{n_layers-1}\n{'─'*65}")
        start = datetime.now()

        # Check if vector already exists
        existing = find_vector_file(vec_dir, layer)
        if existing and args.skip_existing:
            vec = torch.load(existing, map_location="cpu")
            print(f"  Skipping — vector exists: {existing}")
            print(f"  shape={vec.shape}  norm={vec.norm().item():.4f}")
            results[layer] = {"status": "skipped", "path": existing}
            continue

        if existing:
            print(f"  Overwriting existing vector: {existing}")

        try:
            train_layer(layer, train_data, config_path, dataset_key)
            path = find_vector_file(vec_dir, layer)
            if not path:
                raise FileNotFoundError(
                    f"Vector not found after training for layer {layer}. "
                    f"Check EasyEdit save path under {vec_dir}."
                )
            vec      = torch.load(path, map_location="cpu")
            duration = (datetime.now() - start).total_seconds() / 60
            print(f"  Saved: {path}")
            print(f"  shape={vec.shape}  norm={vec.norm().item():.4f}  "
                  f"[{duration:.1f} min]")
            results[layer] = {
                "status":       "trained",
                "path":         path,
                "shape":        list(vec.shape),
                "norm":         round(vec.norm().item(), 4),
                "duration_min": round(duration, 2),
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            results[layer] = {"status": "error", "error": str(e)}

    # Summary
    print(f"\n{'='*65}\nSUMMARY\n{'='*65}")
    trained  = [l for l, r in results.items() if r["status"] == "trained"]
    skipped  = [l for l, r in results.items() if r["status"] == "skipped"]
    errors   = [l for l, r in results.items() if r["status"] == "error"]
    print(f"  Trained:  {trained}")
    print(f"  Skipped:  {skipped}")
    print(f"  Errors:   {errors}")

    if errors:
        print(f"\n  WARNING: {len(errors)} layer(s) failed. "
              f"Re-run without --skip-existing to retry.")

    # Save log
    log_dir  = os.path.join(_CODE_DIR, "results",
                             model_name.replace("/", "_"), dataset_key)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_vectors_{timestamp}.json")
    with open(log_file, "w") as f:
        json.dump({
            "timestamp":    datetime.now().isoformat(),
            "config":       config_path,
            "model":        model_name,
            "dataset":      dataset_key,
            "target_layers": target_layers,
            "results":      {str(l): r for l, r in results.items()},
        }, f, indent=2)
    print(f"\n  Log saved: {log_file}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)


if __name__ == "__main__":
    main()