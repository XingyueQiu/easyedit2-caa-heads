#!/usr/bin/env python
# coding: utf-8
"""
scripts/06_mu_sweep.py
======================
Mu sweep — characterises how the greedy best subset performs across a
range of steering multipliers.

Important: the best subset was found at a fixed mu (default 2.0) during
greedy search. This script evaluates that same subset at different mu
values WITHOUT re-running the search. This is characterisation only —
it shows robustness of the subset across mu values, not a new selection.

Evaluates at each mu:
  1. No steering (baseline)   — same score regardless of mu
  2. Default CAA (block-level) — re-evaluated at each mu
  3. All heads attn-level     — re-evaluated at each mu
  4. Greedy best subset       — re-evaluated at each mu

Can run in parallel with 05_random_k.py and 07_per_input.py.

Reads:  results/<model>/<dataset>/layer<N>/greedy_*.json
        results/<model>/<dataset>/layer<N>/baselines_*.json
Writes: results/<model>/<dataset>/layer<N>/mu_sweep_<timestamp>.json

Usage:
    cd ~/EasyEdit_clean
    python code/scripts/06_mu_sweep.py --layer 6
    python code/scripts/06_mu_sweep.py --layer 6 --mu-values -2 -1 0.5 1 1.5 2 2.5 3
    python code/scripts/06_mu_sweep.py --layer 6 --eval sentiment
"""

import os, sys, json, argparse, glob
import torch
import yaml
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

DEFAULT_CONFIG  = os.path.join(_CODE_DIR, "configs", "config_toxicity-gpt2.yaml")
GENERATE_YAML   = os.path.join(_PROJECT_DIR, "hparams", "Steer", "caa_hparams", "generate_caa.yaml")
APPLY_YAML      = os.path.join(_PROJECT_DIR, "hparams", "Steer", "caa_hparams", "apply_caa.yaml")

SPLIT_SEED        = 42
TRAIN_VECTOR_FRAC = 0.80
EVAL_NUM_SAMPLES  = 200
GENERATION_SEED   = 42

DEFAULT_MU_VALUES = [-2.0, -1.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


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
        description="Mu sweep characterisation of greedy best subset")
    parser.add_argument("--layer",     type=int, required=True)
    parser.add_argument("--config",    default=DEFAULT_CONFIG)
    parser.add_argument("--eval",      default="toxigen",
                        choices=["toxigen", "sentiment"])
    parser.add_argument("--mu-values", nargs="+", type=float,
                        default=DEFAULT_MU_VALUES,
                        help="Mu values to sweep")
    parser.add_argument("--samples",   type=int, default=EVAL_NUM_SAMPLES)
    args = parser.parse_args()

    layer        = args.layer
    config_path  = args.config
    eval_method  = args.eval
    mu_values    = sorted(args.mu_values)
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
    results_file = os.path.join(results_dir, "mu_sweep_tmp.json")

    print("=" * 65)
    print("MU SWEEP")
    print(f"Config:     {config_path}")
    print(f"Model:      {model_name}")
    print(f"Dataset:    {dataset_key}")
    print(f"Layer:      {layer}")
    print(f"Eval:       {eval_method}")
    print(f"Mu values:  {mu_values}")
    print(f"Samples:    {eval_samples}")
    print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print("\n  NOTE: best subset was found at fixed mu during greedy search.")
    print("  This sweep characterises robustness — not a new selection.")

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
    search_mu    = greedy["mu"]

    print(f"\n  Greedy best subset: {best_subset}  (found at mu={search_mu})")

    # Load baseline score from baselines file — no steering, score is
    # independent of mu so we only need it once
    baseline_files = sorted(glob.glob(
        os.path.join(results_dir, "baselines_layer*.json")))
    baseline_score = None
    if baseline_files:
        with open(baseline_files[-1]) as f:
            bl = json.load(f)
        baseline_score = bl["scores"]["no_steering"]
        print(f"  Baseline (no steering): {baseline_score:.4f}%  "
              f"(loaded from baselines file)")

    # ── Data split ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}\nDATA SPLIT\n{'='*65}")
    _, eval_pool = make_splits(
        cfg, dataset_key=dataset_key,
        train_frac=TRAIN_VECTOR_FRAC,
        eval_cap=eval_samples,
        seed=SPLIT_SEED,
    )

    # If baseline not loaded from file, evaluate it once (mu-independent)
    if baseline_score is None:
        print(f"\n  Evaluating baseline (no steering) ...")
        baseline_score = _eval_baseline(
            cfg, model_name, eval_pool, results_file, dataset_key, eval_method
        )
        print(f"  Baseline: {baseline_score:.4f}%")

    # ── Mu sweep ─────────────────────────────────────────────────────────────
    print(f"\n{'='*65}\nSWEEPING {len(mu_values)} MU VALUES\n{'='*65}")

    table = {}

    for mu in mu_values:
        print(f"\n  mu={mu}")

        # Default CAA (block-level)
        set_apply_yaml(APPLY_YAML, GENERATE_YAML, layer, mu,
                       use_head_steering=False)
        default_caa = _eval_caa(cfg, model_name, eval_pool,
                                 results_file, dataset_key, eval_method)

        # All heads attn-level
        set_apply_yaml(APPLY_YAML, GENERATE_YAML, layer, mu,
                       use_head_steering=True,
                       attention_heads=list(range(n_heads)))
        all_heads = _eval_caa(cfg, model_name, eval_pool,
                               results_file, dataset_key, eval_method)

        # Greedy best subset (attn-level, fixed subset from search)
        set_apply_yaml(APPLY_YAML, GENERATE_YAML, layer, mu,
                       use_head_steering=True,
                       attention_heads=best_subset)
        best = _eval_caa(cfg, model_name, eval_pool,
                          results_file, dataset_key, eval_method)

        table[mu] = {
            "baseline":    baseline_score,
            "default_caa": round(default_caa, 4),
            "all_heads":   round(all_heads, 4),
            "best_subset": round(best, 4),
            "vs_default_caa": round(best - default_caa, 4),
            "vs_baseline":    round(best - baseline_score, 4),
        }
        search_marker = " <-- search mu" if mu == search_mu else ""
        print(f"    baseline={baseline_score:.2f}%  "
              f"default_caa={default_caa:.2f}%  "
              f"all_heads={all_heads:.2f}%  "
              f"best_subset={best:.2f}%  "
              f"(vs CAA: {best-default_caa:+.2f}%){search_marker}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}\nSUMMARY  (layer {layer}, subset={best_subset})\n{'='*65}")
    print(f"  {'mu':>6} | {'Baseline':>9} | {'Def CAA':>9} | "
          f"{'All Heads':>10} | {'Best Sub':>9} | {'vs CAA':>7}")
    print(f"  {'─'*6}-+-{'─'*9}-+-{'─'*9}-+-{'─'*10}-+-{'─'*9}-+-{'─'*7}")
    for mu in mu_values:
        r = table[mu]
        marker = "*" if mu == search_mu else " "
        print(f"  {mu:>6}{marker}| {r['baseline']:>9.4f}% | "
              f"{r['default_caa']:>9.4f}% | "
              f"{r['all_heads']:>10.4f}% | "
              f"{r['best_subset']:>9.4f}% | "
              f"{r['vs_default_caa']:>+7.4f}%")
    print(f"\n  * = mu used during greedy search")

    # Assess consistency — does best subset beat default CAA across most mu?
    beats_caa = sum(1 for r in table.values() if r["vs_default_caa"] > 0)
    print(f"\n  Best subset beats default CAA at "
          f"{beats_caa}/{len(mu_values)} mu values.")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_file = os.path.join(results_dir,
                             f"mu_sweep_layer{layer}_{timestamp}.json")
    output = {
        "timestamp":    datetime.now().isoformat(),
        "config":       config_path,
        "model":        model_name,
        "dataset":      dataset_key,
        "eval_method":  eval_method,
        "layer":        layer,
        "n_heads":      n_heads,
        "search_mu":    search_mu,
        "mu_values":    mu_values,
        "eval_samples": eval_samples,
        "best_subset":  best_subset,
        "baseline":     baseline_score,
        "table":        {str(mu): v for mu, v in table.items()},
        "beats_caa_count": beats_caa,
        "beats_caa_total": len(mu_values),
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
