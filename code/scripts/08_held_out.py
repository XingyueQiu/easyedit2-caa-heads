#!/usr/bin/env python
# coding: utf-8
"""
scripts/08_held_out.py
======================
Final held-out evaluation on dev.jsonl — the clean test of whether
the greedy best subset generalises to unseen data.

Evaluates on dev.jsonl (never touched during vector training or search):
  1. No steering       — baseline
  2. Default CAA       — block-level, standard reference
  3. All heads attn    — attention-level, all heads
  4. Greedy best subset — the subset found by 04_greedy_search.py

This is the primary evidence for RQ1:
  "Does head-level steering provide more effective control than block-level CAA?"

The answer comes from comparing row 4 against row 2 on held-out data.

Reads:  results/<model>/<dataset>/layer<N>/greedy_*.json
        data/dev.jsonl
Writes: results/<model>/<dataset>/layer<N>/held_out_<timestamp>.json

Usage:
    cd ~/EasyEdit_clean
    python code/scripts/08_held_out.py --layer 6
    python code/scripts/08_held_out.py --layer 6 --dev data/dev.jsonl
    python code/scripts/08_held_out.py --layer 6 --eval sentiment
"""

import os, sys, json, argparse, glob
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
    load_jsonl, format_pool,
    eval_baseline, eval_caa,
)

# ============================================================================
# Defaults
# ============================================================================

DEFAULT_CONFIG  = os.path.join(_CODE_DIR, "configs", "config_toxicity-gpt2.yaml")
DEFAULT_DEV     = None   # loaded from dataset_format.yaml via --dev-dataset
GENERATE_YAML   = os.path.join(_PROJECT_DIR, "hparams", "Steer", "caa_hparams", "generate_caa.yaml")
APPLY_YAML      = os.path.join(_PROJECT_DIR, "hparams", "Steer", "caa_hparams", "apply_caa.yaml")

GENERATION_SEED = 42
HOLDOUT_CAP     = 200   # max samples from dev.jsonl


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
    parser = argparse.ArgumentParser(
        description="Final held-out evaluation on dev.jsonl")
    parser.add_argument("--layer",   type=int, required=True)
    parser.add_argument("--config",  default=DEFAULT_CONFIG)
    parser.add_argument("--dev",     default=DEFAULT_DEV,
                        help="Path to dev.jsonl (overrides --dev-dataset)")
    parser.add_argument("--dev-dataset", default="toxicity_val1",
                        help="Dataset name in dataset_format.yaml to use as dev set")
    parser.add_argument("--eval",    default="toxigen",
                        choices=["toxigen", "sentiment"])
    parser.add_argument("--mu",      type=float, default=None,
                        help="Override mu (default: use mu from greedy search)")
    parser.add_argument("--cap",     type=int,   default=None,
                        help="Max samples from dev set (default: all)")
    args = parser.parse_args()

    layer        = args.layer
    config_path  = args.config
    dev_path     = args.dev
    eval_method  = args.eval
    mu_override  = args.mu
    holdout_cap  = args.cap
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
    results_file = os.path.join(results_dir, "held_out_tmp.json")

    print("=" * 65)
    print("HELD-OUT EVALUATION  (RQ1 primary evidence)")
    print(f"Config:   {config_path}")
    print(f"Model:    {model_name}")
    print(f"Dataset:  {dataset_key}")
    print(f"Layer:    {layer}")
    print(f"Eval:     {eval_method}")
    print(f"Dev:      {dev_path}")
    print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Pre-flight ───────────────────────────────────────────────────────────
    if not find_vector_file(vec_dir, layer):
        print(f"\n  ERROR: no vector for layer {layer}.")
        print(f"  Run: python code/scripts/00_train_vectors.py --layers {layer}")
        return

    greedy_files = sorted(glob.glob(
        os.path.join(results_dir, "greedy_layer*.json")))
    if not greedy_files:
        print(f"\n  ERROR: no greedy results in {results_dir}")
        print(f"  Run: python code/scripts/04_greedy_search.py --layer {layer}")
        return

    with open(greedy_files[-1]) as f:
        greedy = json.load(f)

    best_subset  = greedy["best_subset"]
    greedy_k     = len(best_subset)
    search_mu    = greedy["mu"]
    mu           = mu_override if mu_override is not None else search_mu

    # Load search-time scores for reference
    search_baseline    = greedy["baselines"]["no_steering"]
    search_default_caa = greedy["baselines"]["default_caa"]
    search_best        = greedy["best_score"]

    print(f"\n  Greedy best subset: {best_subset}  K={greedy_k}")
    print(f"  MU: {mu}  (search mu was {search_mu})")
    print(f"\n  Search-time scores (val pool, for reference):")
    print(f"    Baseline:    {search_baseline:.4f}%")
    print(f"    Default CAA: {search_default_caa:.4f}%")
    print(f"    Best subset: {search_best:.4f}%")

    # ── Load dev set ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}\nDEV SET\n{'='*65}")
    dev_path = args.dev
    dev_dataset = args.dev_dataset

    if dev_path is not None:
        # Explicit path provided
        dev_path_abs = os.path.abspath(dev_path)
        if not os.path.exists(dev_path_abs):
            print(f"\n  ERROR: dev file not found: {dev_path_abs}")
            return
        dev_pool = load_jsonl(dev_path_abs, cap=holdout_cap if holdout_cap else None)
        print(f"  Loaded {len(dev_pool)} samples from {dev_path_abs}")
    else:
        # Load from dataset_format.yaml via DatasetLoader
        from steer.datasets.dataset_loader import DatasetLoader
        loader   = DatasetLoader()
        dev_pool = loader.load_file(dataset_name=dev_dataset, split="generation")
        if holdout_cap:
            dev_pool = dev_pool[:holdout_cap]
        print(f"  Loaded {len(dev_pool)} samples from dataset '{dev_dataset}' "
              f"(dataset_format.yaml)")
    print(f"  These samples were NEVER used during vector training or search.")

    # ── Evaluation ───────────────────────────────────────────────────────────
    scores = {}

    # 1. No steering
    print(f"\n{'='*65}\n1. No steering\n{'='*65}")
    no_steer = _eval_baseline(cfg, model_name, dev_pool,
                               results_file, dataset_key, eval_method)
    scores["no_steering"] = round(no_steer, 4)
    print(f"  No steering: {no_steer:.4f}%")

    # 2. Default CAA (block-level)
    print(f"\n{'='*65}\n2. Default CAA (block-level)\n{'='*65}")
    set_apply_yaml(APPLY_YAML, GENERATE_YAML, layer, mu,
                   use_head_steering=False)
    default_caa = _eval_caa(cfg, model_name, dev_pool,
                             results_file, dataset_key, eval_method)
    scores["default_caa"] = round(default_caa, 4)
    print(f"  Default CAA: {default_caa:.4f}%  "
          f"(vs no steering: {default_caa-no_steer:+.4f}%)")

    # 3. All heads attn-level
    print(f"\n{'='*65}\n3. All heads, attention-level\n{'='*65}")
    set_apply_yaml(APPLY_YAML, GENERATE_YAML, layer, mu,
                   use_head_steering=True,
                   attention_heads=list(range(n_heads)))
    all_heads = _eval_caa(cfg, model_name, dev_pool,
                           results_file, dataset_key, eval_method)
    scores["all_heads_attn"] = round(all_heads, 4)
    print(f"  All heads (attn): {all_heads:.4f}%  "
          f"(vs no steering: {all_heads-no_steer:+.4f}%)")

    # 4. Greedy best subset
    print(f"\n{'='*65}\n4. Greedy best subset {best_subset}\n{'='*65}")
    set_apply_yaml(APPLY_YAML, GENERATE_YAML, layer, mu,
                   use_head_steering=True,
                   attention_heads=best_subset)
    best = _eval_caa(cfg, model_name, dev_pool,
                      results_file, dataset_key, eval_method)
    scores["greedy_subset"] = round(best, 4)
    print(f"  Greedy subset: {best:.4f}%  "
          f"(vs no steering: {best-no_steer:+.4f}%)")

    # ── RQ1 summary ───────────────────────────────────────────────────────────
    vs_default_caa  = best - default_caa
    vs_all_heads    = best - all_heads
    caa_beats_base  = default_caa - no_steer
    best_beats_base = best - no_steer

    print(f"\n{'='*65}\nRQ1 SUMMARY  (held-out dev set)\n{'='*65}")
    print(f"\n  {'Method':<25} {'Score':>8} {'vs Baseline':>12} {'vs Def CAA':>11}")
    print(f"  {'─'*25}-+-{'─'*8}-+-{'─'*12}-+-{'─'*11}")
    print(f"  {'No steering':<25} {no_steer:>8.4f}%  {'—':>12}  {'—':>11}")
    print(f"  {'Default CAA (block)':<25} {default_caa:>8.4f}%  "
          f"{caa_beats_base:>+12.4f}%  {'—':>11}")
    print(f"  {'All heads (attn-level)':<25} {all_heads:>8.4f}%  "
          f"{all_heads-no_steer:>+12.4f}%  "
          f"{all_heads-default_caa:>+11.4f}%")
    print(f"  {'Greedy subset (K='+str(greedy_k)+')':<25} {best:>8.4f}%  "
          f"{best_beats_base:>+12.4f}%  "
          f"{vs_default_caa:>+11.4f}%")

    print(f"\n  Greedy vs default CAA:  {vs_default_caa:+.4f}%  "
          f"({'BETTER' if vs_default_caa > 0 else 'WORSE'} on held-out)")
    print(f"  Greedy vs all heads:    {vs_all_heads:+.4f}%")
    print(f"\n  Search-time vs held-out generalisation:")
    print(f"    Search greedy score:  {search_best:.4f}%")
    print(f"    Held-out greedy score: {best:.4f}%")
    print(f"    Delta:                {best-search_best:+.4f}%  "
          f"({'generalises well' if abs(best-search_best) < 2.0 else 'some overfitting'})")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_file = os.path.join(results_dir,
                             f"held_out_layer{layer}_{timestamp}.json")
    output = {
        "timestamp":       datetime.now().isoformat(),
        "config":          config_path,
        "model":           model_name,
        "dataset":         dataset_key,
        "eval_method":     eval_method,
        "layer":           layer,
        "n_heads":         n_heads,
        "mu":              mu,
        "dev_source":      dev_path if dev_path else dev_dataset,
        "dev_samples":     len(dev_pool),
        "best_subset":     best_subset,
        "greedy_k":        greedy_k,
        "scores":          scores,
        "vs_default_caa":  round(vs_default_caa, 4),
        "vs_all_heads":    round(vs_all_heads, 4),
        "vs_baseline":     round(best_beats_base, 4),
        "rq1_verdict":     "head-level better" if vs_default_caa > 0 else "head-level worse",
        "generalisation":  {
            "search_score":   search_best,
            "held_out_score": round(best, 4),
            "delta":          round(best - search_best, 4),
        },
        "search_time_reference": {
            "no_steering": search_baseline,
            "default_caa": search_default_caa,
            "best_subset": search_best,
        },
    }
    import numpy as np
    def _json_safe(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Not JSON serialisable: {type(obj)}")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=_json_safe)
    print(f"\n  Results saved: {out_file}")

    if os.path.exists(results_file):
        os.remove(results_file)

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)


if __name__ == "__main__":
    main()