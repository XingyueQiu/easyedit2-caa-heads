#!/usr/bin/env python
# coding: utf-8
"""
scripts/03_baselines.py
=======================
Evaluate all baselines for a given model and target layer.

Baselines:
  1. No steering          — raw model output, no CAA applied
  2. Default CAA          — block-level steering (after FFN), standard EasyEdit2
  3. All heads, attn-level — head-level API, all heads, injected before FFN
  4. All heads, block-level — head-level API, all heads, injected after FFN
                              (same injection point as default CAA)

  Comparison of 3 vs 4 justifies injection point choice for greedy search.
  Attention-level (3) is theoretically preferred: intervenes directly on
  head outputs before FFN mixing, consistent with the hypothesis that
  specific heads are mechanistically responsible for the target behaviour.

These scores are the comparison anchors for RQ1 and RQ2:
  - Greedy best subset is compared against default CAA (RQ1)
  - Baselines 3 vs 4 justify injection point: attention-level is used
    throughout greedy search as it targets head outputs before FFN mixing

Reads:  nothing (self-contained)
Writes: results/<model>/<dataset>/baselines_layer<N>_<timestamp>.json

Usage:
    cd ~/EasyEdit_clean
    python code/scripts/03_baselines.py --layer 6
    python code/scripts/03_baselines.py --layer 6 --config code/configs/config_toxicity-gpt2.yaml
    python code/scripts/03_baselines.py --layer 6 --eval sentiment
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
    load_config, first_path,
    get_n_heads, set_apply_yaml, find_vector_file,
    make_splits, format_pool,
    eval_baseline, eval_caa,
)

# ============================================================================
# Defaults
# ============================================================================

DEFAULT_CONFIG = os.path.join(_CODE_DIR, "configs", "config_toxicity-gpt2.yaml")
GENERATE_YAML  = os.path.join(_PROJECT_DIR, "hparams", "Steer", "caa_hparams", "generate_caa.yaml")
APPLY_YAML     = os.path.join(_PROJECT_DIR, "hparams", "Steer", "caa_hparams", "apply_caa.yaml")

SPLIT_SEED        = 42
TRAIN_VECTOR_FRAC = 0.80
EVAL_NUM_SAMPLES  = 200
GENERATION_SEED   = 42
MU                = 2.0


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
    parser = argparse.ArgumentParser(description="Evaluate baselines for a target layer")
    parser.add_argument("--layer",   type=int, required=True,
                        help="Target layer (from 01_layer_sweep.py recommendation)")
    parser.add_argument("--config",  default=DEFAULT_CONFIG)
    parser.add_argument("--eval",    default="toxigen",
                        choices=["toxigen", "sentiment"])
    parser.add_argument("--mu",      type=float, default=MU)
    parser.add_argument("--samples", type=int,   default=EVAL_NUM_SAMPLES)
    args = parser.parse_args()

    layer        = args.layer
    config_path  = args.config
    eval_method  = args.eval
    mu           = args.mu
    eval_samples = args.samples
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Setup ────────────────────────────────────────────────────────────────
    cfg         = load_config(config_path)
    from omegaconf import OmegaConf
    cfg_plain   = OmegaConf.to_container(cfg, resolve=True)
    model_name  = str(cfg_plain.get("model_name_or_path", "gpt2"))
    vec_dir     = first_path(cfg_plain.get("steer_vector_load_dir"))
    dataset_key = cfg_plain.get("steer_train_dataset", ["toxicity"])
    if isinstance(dataset_key, list):
        dataset_key = str(dataset_key[0])
    else:
        dataset_key = str(dataset_key)
    n_heads     = get_n_heads(model_name)

    results_dir  = os.path.join(_CODE_DIR, "results",
                                 model_name.replace("/", "_"),
                                 dataset_key, f"layer{layer}")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "baselines_tmp.json")

    print("=" * 65)
    print("BASELINES")
    print(f"Config:   {config_path}")
    print(f"Model:    {model_name}")
    print(f"Dataset:  {dataset_key}")
    print(f"Layer:    {layer}")
    print(f"Eval:     {eval_method}")
    print(f"MU:       {mu}")
    print(f"Samples:  {eval_samples}")
    print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Pre-flight ───────────────────────────────────────────────────────────
    vec_path = find_vector_file(vec_dir, layer)
    if not vec_path:
        print(f"\n  ERROR: no vector found for layer {layer} under {vec_dir}")
        print(f"  Run: python code/scripts/00_train_vectors.py "
              f"--config {config_path} --layers {layer}")
        return
    print(f"\n  Vector: {vec_path}")

    # ── Data split ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}\nDATA SPLIT\n{'='*65}")
    _, eval_pool = make_splits(
        cfg, dataset_key=dataset_key,
        train_frac=TRAIN_VECTOR_FRAC,
        eval_cap=eval_samples,
        seed=SPLIT_SEED,
    )

    scores = {}

    # ── Baseline 1: No steering ──────────────────────────────────────────────
    print(f"\n{'='*65}\nBASELINE 1 — No steering\n{'='*65}")
    no_steer = _eval_baseline(cfg, model_name, eval_pool,
                               results_file, dataset_key, eval_method)
    scores["no_steering"] = float(no_steer)
    print(f"  No steering: {no_steer:.4f}%")

    # ── Baseline 2: Default CAA (block-level) ────────────────────────────────
    print(f"\n{'='*65}\nBASELINE 2 — Default CAA (block-level)\n{'='*65}")
    set_apply_yaml(APPLY_YAML, GENERATE_YAML, layer, mu,
                   use_head_steering=False)
    default_caa = _eval_caa(cfg, model_name, eval_pool,
                             results_file, dataset_key, eval_method)
    scores["default_caa"] = float(default_caa)
    print(f"  Default CAA: {default_caa:.4f}%  "
          f"(vs no steering: {default_caa-no_steer:+.4f}%)")

    # ── Baseline 3: All heads explicit (attention-level) ───────────────────
    # Head-level API with all heads selected, injected at attention output
    # (before FFN). This is intentionally different from default CAA which
    # injects at block output (after FFN). The score difference between
    # Baseline 2 and 3 isolates the effect of injection point alone,
    # independent of head selection. Greedy subset (script 04) then shows
    # whether selective attention-level injection can exceed both.
    print(f"\n{'='*65}\nBASELINE 3 — All heads explicit (attention-level)\n{'='*65}")
    set_apply_yaml(APPLY_YAML, GENERATE_YAML, layer, mu,
                   use_head_steering=True,
                   attention_heads=list(range(n_heads)))
    all_heads = _eval_caa(cfg, model_name, eval_pool,
                           results_file, dataset_key, eval_method)
    scores["all_heads_explicit"] = float(all_heads)
    print(f"  All heads explicit: {all_heads:.4f}%  "
          f"(vs default CAA: {all_heads-default_caa:+.4f}%)")
    print(f"  Difference reflects injection point: attention-level vs block-level.")

    # ── Baseline 4: All heads, block-level ──────────────────────────────────
    # Same as Baseline 3 but injected at block output (after FFN).
    # Comparing 3 vs 4 isolates injection point with head selection held fixed.
    print(f"\n{'='*65}\nBASELINE 4 — All heads, block-level injection\n{'='*65}")
    set_apply_yaml(APPLY_YAML, GENERATE_YAML, layer, mu,
                   use_head_steering=True,
                   attention_heads=list(range(n_heads)))
    # Switch injection_level to block in apply yaml directly
    import yaml as _yaml
    with open(APPLY_YAML) as f: _a = _yaml.safe_load(f)
    _a["injection_level"] = "block"
    with open(APPLY_YAML, "w") as f: _yaml.dump(_a, f)
    all_heads_block = _eval_caa(cfg, model_name, eval_pool,
                                 results_file, dataset_key, eval_method)
    scores["all_heads_block_level"] = float(all_heads_block)
    print(f"  All heads block-level: {all_heads_block:.4f}%  "
          f"(vs attn-level: {all_heads-all_heads_block:+.4f}%)")
    # Restore injection_level to attention for subsequent scripts
    with open(APPLY_YAML) as f: _a = _yaml.safe_load(f)
    _a["injection_level"] = "attention"
    with open(APPLY_YAML, "w") as f: _yaml.dump(_a, f)

    injection_diff = all_heads - all_heads_block
    preferred      = "attention-level" if injection_diff >= 0 else "block-level"
    print(f"  Injection point comparison: attention={all_heads:.4f}%  "
          f"block={all_heads_block:.4f}%  diff={injection_diff:+.4f}%")
    print(f"  Preferred by score: {preferred}")
    print(f"  Note: attention-level is theoretically preferred regardless —")
    print(f"  it targets head outputs before FFN mixing.")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}\nSUMMARY  (layer {layer})\n{'='*65}")
    print(f"  No steering:             {no_steer:.4f}%")
    print(f"  Default CAA (block):     {default_caa:.4f}%  "
          f"({default_caa-no_steer:+.4f}% vs no steering)")
    print(f"  All heads attn-level:    {all_heads:.4f}%  "
          f"({all_heads-no_steer:+.4f}% vs no steering)")
    print(f"  All heads block-level:   {all_heads_block:.4f}%  "
          f"({all_heads_block-no_steer:+.4f}% vs no steering)")
    print(f"\n  Injection point diff (attn - block): {injection_diff:+.4f}%")
    print(f"  CAA effect at layer {layer}: {default_caa-no_steer:+.4f}%")
    print(f"  This is the improvement your greedy search must beat (RQ1).")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_file = os.path.join(results_dir,
                             f"baselines_layer{layer}_{timestamp}.json")
    output = {
        "timestamp":         datetime.now().isoformat(),
        "config":            config_path,
        "model":             model_name,
        "dataset":           dataset_key,
        "eval_method":       eval_method,
        "layer":             layer,
        "n_heads":           n_heads,
        "mu":                mu,
        "eval_samples":      eval_samples,
        "vec_path":          vec_path,
        "scores":            scores,
        "caa_improvement":   round(default_caa - no_steer, 4),
        "injection_point_diff":    round(all_heads - all_heads_block, 4),
        "preferred_injection":     "attention" if all_heads >= all_heads_block else "block",
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {out_file}")

    # Cleanup
    if os.path.exists(results_file):
        os.remove(results_file)

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
