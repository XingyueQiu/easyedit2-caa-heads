#!/usr/bin/env python
# coding: utf-8
"""
scripts/01_layer_sweep.py
=========================
Layer sweep for CAA layer selection. Assumes vectors are already trained
by 00_train_vectors.py.

For each layer:
  1. Evaluate baseline (once, reused)
  2. Evaluate default CAA (block-level)
  3. Evaluate each single head independently
  4. Compute dominance score

Layer selection uses a two-stage approach:
  Stage 1: keep only layers where CAA improvement > IMPROVEMENT_THRESHOLD
  Stage 2: among those, pick the layer with lowest dominance
           (most distributed head contributions — most informative for RQ2-4)

Usage:
    cd ~/EasyEdit_clean
    python code/scripts/01_layer_sweep.py
    python code/scripts/01_layer_sweep.py --config code/configs/config_toxicity-gpt2.yaml
    python code/scripts/01_layer_sweep.py --config code/configs/config_sst2-gpt2.yaml --eval sentiment
    python code/scripts/01_layer_sweep.py --layers 2 6 10   # specific layers only
"""

import os, sys, json, statistics, argparse
import torch
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR    = os.path.dirname(_SCRIPT_DIR)
_PROJECT_DIR = os.path.dirname(_CODE_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _CODE_DIR)

from utils import (
    load_config, first_path,
    get_n_layers, get_n_heads,
    set_apply_yaml, find_vector_file,
    make_splits, format_pool,
    eval_baseline, eval_caa,
)

# ============================================================================
# Defaults
# ============================================================================

DEFAULT_CONFIG        = os.path.join(_CODE_DIR, "configs", "config_toxicity-gpt2.yaml")
GENERATE_YAML         = os.path.join(_PROJECT_DIR, "hparams", "Steer", "caa_hparams", "generate_caa.yaml")
APPLY_YAML            = os.path.join(_PROJECT_DIR, "hparams", "Steer", "caa_hparams", "apply_caa.yaml")

SPLIT_SEED            = 42
TRAIN_VECTOR_FRAC     = 0.80
EVAL_NUM_SAMPLES      = 200
GENERATION_SEED       = 42
MU                    = 2.0
IMPROVEMENT_THRESHOLD = 2.0   # min pp improvement over baseline for stage-1


# ============================================================================
# Eval wrappers
# ============================================================================

def _fmt(pool, dataset_key):
    return format_pool(pool, dataset_key=dataset_key, use_negative_prompt=True)


def _eval_baseline(cfg, model_name, pool, results_file, dataset_key, eval_method):
    return eval_baseline(
        cfg=cfg, model_name=model_name, pool=pool,
        results_file=results_file,
        format_pool_fn=lambda p: _fmt(p, dataset_key),
        dataset_key=dataset_key,
        generation_seed=GENERATION_SEED,
        eval_method=eval_method,
    )


def _eval_caa(cfg, model_name, pool, results_file, dataset_key, eval_method):
    return eval_caa(
        cfg=cfg, pool=pool, results_file=results_file,
        format_pool_fn=lambda p: _fmt(p, dataset_key),
        dataset_key=dataset_key,
        model_name=model_name,
        eval_method=eval_method,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Layer sweep for CAA layer selection")
    parser.add_argument("--config",    default=DEFAULT_CONFIG)
    parser.add_argument("--eval",      default="toxigen",
                        choices=["toxigen", "sentiment"])
    parser.add_argument("--mu",        type=float, default=MU)
    parser.add_argument("--samples",   type=int,   default=EVAL_NUM_SAMPLES)
    parser.add_argument("--threshold", type=float, default=IMPROVEMENT_THRESHOLD,
                        help="Min pp improvement over baseline for layer candidacy")
    parser.add_argument("--layers",    nargs="+",  type=int, default=None,
                        help="Specific layers to evaluate (default: all)")
    args = parser.parse_args()

    config_path   = args.config
    eval_method   = args.eval
    mu            = args.mu
    eval_samples  = args.samples
    threshold     = args.threshold
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Setup ────────────────────────────────────────────────────────────────
    cfg         = load_config(config_path)
    model_name  = cfg.get("model_name_or_path", "gpt2")
    vec_dir     = first_path(cfg.get("steer_vector_load_dir"))
    dataset_key = cfg.get("steer_train_dataset", ["toxicity"])
    if isinstance(dataset_key, list):
        dataset_key = dataset_key[0]
    n_layers      = get_n_layers(model_name)
    n_heads       = get_n_heads(model_name)
    target_layers = args.layers if args.layers else list(range(n_layers))

    results_dir  = os.path.join(_CODE_DIR, "results",
                                 model_name.replace("/", "_"), dataset_key)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "layer_sweep_tmp.json")

    print("=" * 65)
    print("LAYER SWEEP")
    print(f"Config:     {config_path}")
    print(f"Model:      {model_name}")
    print(f"Dataset:    {dataset_key}")
    print(f"Eval:       {eval_method}")
    print(f"MU:         {mu}")
    print(f"Samples:    {eval_samples}")
    print(f"Threshold:  >{threshold}pp above baseline")
    print(f"Layers:     {target_layers}")
    print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print(f"\nN layers: {n_layers}  |  N heads: {n_heads}")
    print(f"Vec dir:  {vec_dir}")

    # ── Pre-flight: verify all vectors exist ─────────────────────────────────
    print(f"\n{'='*65}\nVECTOR PRE-FLIGHT CHECK\n{'='*65}")
    missing = []
    for layer in target_layers:
        path = find_vector_file(vec_dir, layer)
        if path:
            print(f"  Layer {layer:>3}: OK  ({path})")
        else:
            print(f"  Layer {layer:>3}: MISSING")
            missing.append(layer)
    if missing:
        print(f"\n  ERROR: vectors missing for layers {missing}.")
        print(f"  Run 00_train_vectors.py first:")
        print(f"  python code/scripts/00_train_vectors.py --config {config_path}")
        return

    # ── Data split ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}\nDATA SPLIT\n{'='*65}")
    _, eval_pool = make_splits(
        cfg, dataset_key=dataset_key,
        train_frac=TRAIN_VECTOR_FRAC,
        eval_cap=eval_samples,
        seed=SPLIT_SEED,
    )

    # ── Baseline ─────────────────────────────────────────────────────────────
    print(f"\n{'='*65}\nBASELINE (no steering)\n{'='*65}")
    baseline = _eval_baseline(cfg, model_name, eval_pool,
                               results_file, dataset_key, eval_method)
    print(f"  Baseline: {baseline:.4f}%")

    # ── Layer sweep ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}\nSWEEPING {len(target_layers)} LAYERS\n{'='*65}")
    layer_results = {}

    for layer in target_layers:
        print(f"\n{'─'*65}\nLayer {layer}/{n_layers-1}\n{'─'*65}")
        start_time = datetime.now()

        try:
            # Default CAA
            set_apply_yaml(APPLY_YAML, GENERATE_YAML, layer, mu,
                           use_head_steering=False)
            caa_score = _eval_caa(cfg, model_name, eval_pool,
                                   results_file, dataset_key, eval_method)
            print(f"  Default CAA: {caa_score:.4f}%  "
                  f"(vs baseline: {caa_score-baseline:+.4f}%)")

            # Single head scores
            single_scores = []
            for h in range(n_heads):
                set_apply_yaml(APPLY_YAML, GENERATE_YAML, layer, mu,
                               use_head_steering=True, attention_heads=[h])
                s = _eval_caa(cfg, model_name, eval_pool,
                               results_file, dataset_key, eval_method)
                single_scores.append(s)
            print(f"  Single heads: {[round(s, 2) for s in single_scores]}")

            improvement = caa_score - baseline
            dominance   = max(single_scores) - statistics.mean(single_scores)
            duration    = (datetime.now() - start_time).total_seconds() / 60

            layer_results[layer] = {
                "default_caa":        float(caa_score),
                "baseline":           float(baseline),
                "improvement":        float(improvement),
                "single_head_scores": [round(s, 4) for s in single_scores],
                "single_head_max":    round(max(single_scores), 4),
                "single_head_mean":   round(statistics.mean(single_scores), 4),
                "dominance":          round(dominance, 4),
                "duration_min":       round(duration, 2),
            }
            print(f"  dominance={dominance:.4f}%  [{duration:.1f} min]")

        except Exception as e:
            print(f"  ERROR evaluating layer {layer}: {e}")
            layer_results[layer] = {
                "default_caa": None, "baseline": float(baseline),
                "improvement": None, "error": str(e), "duration_min": None,
            }

    # ── Two-stage layer selection ─────────────────────────────────────────────
    valid = {l: r for l, r in layer_results.items()
             if r["default_caa"] is not None}

    if not valid:
        print("\nERROR: No layers evaluated successfully.")
        return

    # Stage 1: filter by improvement threshold
    candidates = {l: r for l, r in valid.items()
                  if r["improvement"] > threshold}
    if not candidates:
        print(f"\n  WARNING: No layer exceeds {threshold}pp threshold. "
              f"Falling back to best CAA score.")
        candidates       = valid
        selection_method = "fallback_best_caa"
    else:
        selection_method = f"lowest_dominance_above_{threshold}pp"

    # Stage 2: lowest dominance among candidates
    best_layer = min(candidates, key=lambda l: candidates[l]["dominance"])
    best_score = valid[best_layer]["default_caa"]

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}\nLAYER SWEEP SUMMARY\n{'='*65}")
    print(f"  Baseline:  {baseline:.4f}%")
    print(f"  Threshold: >{threshold}pp improvement required\n")
    print(f"  {'Layer':>6} | {'CAA Score':>10} | {'vs Base':>8} | "
          f"{'Dominance':>10} | {'Candidate':>9} | {'':>8}")
    print(f"  {'─'*6}-+-{'─'*10}-+-{'─'*8}-+-{'─'*10}-+-{'─'*9}-+-{'─'*8}")

    for layer in sorted(layer_results.keys()):
        r = layer_results[layer]
        if r.get("default_caa") is None:
            print(f"  {layer:>6} | {'ERROR':>10} |")
            continue
        in_cand = "yes" if layer in candidates else "no"
        note    = "<-- BEST" if layer == best_layer else ""
        print(f"  {layer:>6} | {r['default_caa']:>10.4f}% | "
              f"{r['improvement']:>+8.4f}% | "
              f"{r['dominance']:>10.4f}% | "
              f"{in_cand:>9} | {note}")

    print(f"\n  Selected layer: {best_layer}")
    print(f"  Method: {selection_method}")
    print(f"  CAA score:  {best_score:.4f}%  "
          f"(+{valid[best_layer]['improvement']:.4f}% vs baseline)")
    print(f"  Dominance:  {valid[best_layer]['dominance']:.4f}%")
    print(f"\n  Next step: set TARGET_LAYERS = [{best_layer}] "
          f"in scripts/04_greedy_search.py")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    model_tag = model_name.replace("/", "_").replace(".", "_")
    out_file  = os.path.join(results_dir,
                              f"layer_sweep_{model_tag}_{timestamp}.json")
    output = {
        "timestamp":             datetime.now().isoformat(),
        "config":                config_path,
        "model":                 model_name,
        "dataset":               dataset_key,
        "eval_method":           eval_method,
        "n_layers":              n_layers,
        "n_heads":               n_heads,
        "mu":                    mu,
        "eval_samples":          eval_samples,
        "improvement_threshold": threshold,
        "baseline":              float(baseline),
        "layer_results":         {str(l): r for l, r in layer_results.items()},
        "best_layer":            best_layer,
        "best_caa_score":        float(best_score),
        "selection_method":      selection_method,
        "recommendation":        f"Use layer {best_layer} for greedy search.",
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {out_file}")

    if os.path.exists(results_file):
        os.remove(results_file)

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
