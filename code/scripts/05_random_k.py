#!/usr/bin/env python
# coding: utf-8
"""
scripts/05_random_k.py
======================
Random-K control experiment for RQ3:
  "Is greedy selection meaningfully better than random head selection?"

For each K from 1 to n_heads:
  - Draw RANDOM_K_REPEATS random subsets of size K (seeded per (K, rep))
  - Evaluate each subset
  - Compare greedy best-K* subset against random at the same K*
  - Run Mann-Whitney U test at K* to assess statistical significance

Can run in parallel with 06_mu_sweep.py and 07_per_input.py after
04_greedy_search.py completes.

Reads:  results/<model>/<dataset>/layer<N>/greedy_*.json
Writes: results/<model>/<dataset>/layer<N>/random_k_<timestamp>.json

Usage:
    cd ~/EasyEdit_clean
    python code/scripts/05_random_k.py --layer 6
    python code/scripts/05_random_k.py --layer 6 --repeats 50
    python code/scripts/05_random_k.py --layer 6 --eval sentiment
"""

import os, sys, json, random, statistics, argparse, glob
import torch
from datetime import datetime
from scipy import stats

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
    eval_caa,
)

# ============================================================================
# Defaults
# ============================================================================

DEFAULT_CONFIG   = os.path.join(_CODE_DIR, "configs", "config_toxicity-gpt2.yaml")
GENERATE_YAML    = os.path.join(_PROJECT_DIR, "hparams", "Steer", "caa_hparams", "generate_caa.yaml")
APPLY_YAML       = os.path.join(_PROJECT_DIR, "hparams", "Steer", "caa_hparams", "apply_caa.yaml")

SPLIT_SEED        = 42
TRAIN_VECTOR_FRAC = 0.80
EVAL_NUM_SAMPLES  = 200
MU                = 2.0
RANDOM_K_REPEATS  = 30


# ============================================================================
# Cache (shared with greedy if run in same process, otherwise fresh)
# ============================================================================

_cache = {}

def cached_eval(cfg, model_name, heads, mu, pool,
                layer, results_file, dataset_key, eval_method, label=""):
    if len(pool) == 1:
        input_id = (pool[0].get("not_matching") or pool[0].get("input", ""))[:80]
    else:
        input_id = None
    key = (tuple(sorted(heads)), mu, layer, input_id)
    if key in _cache:
        return _cache[key]
    set_apply_yaml(APPLY_YAML, GENERATE_YAML, layer, mu,
                   use_head_steering=True, attention_heads=list(heads))
    score = eval_caa(
        cfg=cfg, pool=pool,
        results_file=results_file,
        format_pool_fn=lambda p: format_pool(
            p, dataset_key=dataset_key, use_negative_prompt=True
        ),
        dataset_key=dataset_key,
        model_name=model_name,
        eval_method=eval_method,
    )
    _cache[key] = score
    if label:
        print(f"    {label}: {score:.4f}%")
    return score


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Random-K control for RQ3")
    parser.add_argument("--layer",   type=int, required=True)
    parser.add_argument("--config",  default=DEFAULT_CONFIG)
    parser.add_argument("--eval",    default="toxigen",
                        choices=["toxigen", "sentiment"])
    parser.add_argument("--mu",      type=float, default=MU)
    parser.add_argument("--samples", type=int,   default=EVAL_NUM_SAMPLES)
    parser.add_argument("--repeats", type=int,   default=RANDOM_K_REPEATS,
                        help="Random subsets per K value")
    args = parser.parse_args()

    layer        = args.layer
    config_path  = args.config
    eval_method  = args.eval
    mu           = args.mu
    eval_samples = args.samples
    n_repeats    = args.repeats
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
    results_file = os.path.join(results_dir, "random_k_tmp.json")

    print("=" * 65)
    print("RANDOM-K CONTROL  (RQ3)")
    print(f"Config:   {config_path}")
    print(f"Model:    {model_name}")
    print(f"Dataset:  {dataset_key}")
    print(f"Layer:    {layer}")
    print(f"Eval:     {eval_method}")
    print(f"MU:       {mu}")
    print(f"Samples:  {eval_samples}")
    print(f"Repeats:  {n_repeats} per K")
    print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Pre-flight ───────────────────────────────────────────────────────────
    if not find_vector_file(vec_dir, layer):
        print(f"\n  ERROR: no vector for layer {layer}.")
        print(f"  Run: python code/scripts/00_train_vectors.py --layers {layer}")
        return

    greedy_files = sorted(glob.glob(
        os.path.join(results_dir, "greedy_layer*.json")
    ))
    if not greedy_files:
        print(f"\n  ERROR: no greedy results in {results_dir}")
        print(f"  Run: python code/scripts/04_greedy_search.py --layer {layer}")
        return

    with open(greedy_files[-1]) as f:
        greedy = json.load(f)

    best_subset  = greedy["best_subset"]
    best_score   = greedy["best_score"]
    greedy_k     = len(best_subset)
    baseline     = greedy["baselines"]["no_steering"]
    default_caa  = greedy["baselines"]["default_caa"]

    print(f"\n  Loaded greedy results: {greedy_files[-1]}")
    print(f"  Greedy best subset: {best_subset}  K={greedy_k}")
    print(f"  Greedy best score:  {best_score:.4f}%")
    print(f"  Baseline:           {baseline:.4f}%")
    print(f"  Default CAA:        {default_caa:.4f}%")
    print(f"\n  Note: {n_repeats} repeats per K.")
    print(f"  At K=1: {n_heads} possible subsets — sampling with replacement.")
    from math import comb
    space = comb(n_heads, greedy_k)
    coverage = min(n_repeats / space * 100, 100)
    print(f"  At K={greedy_k}: C({n_heads},{greedy_k})={space} subsets — "
          f"covering ~{coverage:.1f}% of space.")

    # ── Data split ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}\nDATA SPLIT\n{'='*65}")
    _, eval_pool = make_splits(
        cfg, dataset_key=dataset_key,
        train_frac=TRAIN_VECTOR_FRAC,
        eval_cap=eval_samples,
        seed=SPLIT_SEED,
    )

    # ── Random-K sweep ───────────────────────────────────────────────────────
    print(f"\n{'='*65}\nRANDOM-K SWEEP  K=1 to {n_heads}\n{'='*65}")
    random_k_results = {}

    for k in range(1, n_heads + 1):
        print(f"\n  K={k:2d}", end="  ")

        if k == n_heads:
            # Only one possible subset — all heads
            s = cached_eval(cfg, model_name, list(range(n_heads)), mu, eval_pool,
                            layer, results_file, dataset_key, eval_method)
            random_k_results[k] = {
                "scores": [s],
                "mean":   s,
                "std":    0.0,
                "min":    s,
                "max":    s,
                "subsets": [list(range(n_heads))],
                "note":   "only one possible subset at K=n_heads",
            }
            print(f"(full): {s:.4f}%")
            continue

        scores  = []
        subsets = []
        for rep in range(n_repeats):
            # Seed per-(K, rep) — stable across n_repeats changes
            subset = sorted(
                random.Random(SPLIT_SEED + k * 1000 + rep).sample(range(n_heads), k)
            )
            s = cached_eval(cfg, model_name, subset, mu, eval_pool,
                            layer, results_file, dataset_key, eval_method)
            scores.append(s)
            subsets.append(subset)

        mean = statistics.mean(scores)
        std  = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(f"mean={mean:.4f}%  std={std:.4f}%  "
              f"min={min(scores):.4f}%  max={max(scores):.4f}%")

        random_k_results[k] = {
            "scores":  [round(s, 4) for s in scores],
            "mean":    round(mean, 4),
            "std":     round(std, 4),
            "min":     round(min(scores), 4),
            "max":     round(max(scores), 4),
            "subsets": subsets,
        }

    # ── Key comparison: greedy K* vs random K* ───────────────────────────────
    print(f"\n{'='*65}\nKEY COMPARISON: greedy K={greedy_k} vs random K={greedy_k}\n{'='*65}")

    random_at_k = random_k_results[greedy_k]["scores"]
    random_mean = random_k_results[greedy_k]["mean"]
    random_std  = random_k_results[greedy_k]["std"]
    random_max  = random_k_results[greedy_k]["max"]

    # Mann-Whitney U test — non-parametric, appropriate since distribution
    # of random scores may not be normal and n=30 is modest
    # H0: greedy score comes from same distribution as random scores
    # We use a one-sided test: greedy > random
    mw_stat, mw_pval_two = stats.mannwhitneyu(
        [best_score], random_at_k, alternative="two-sided"
    )
    _, mw_pval_one = stats.mannwhitneyu(
        [best_score], random_at_k, alternative="greater"
    )

    # Also run t-test for reference (assumes normality)
    t_stat, t_pval = stats.ttest_1samp(random_at_k, best_score)

    print(f"  Greedy best (K={greedy_k}):  {best_score:.4f}%")
    print(f"  Random mean (K={greedy_k}):  {random_mean:.4f}%  "
          f"std={random_std:.4f}%")
    print(f"  Random max  (K={greedy_k}):  {random_max:.4f}%")
    print(f"  Gap (greedy - random mean):  {best_score-random_mean:+.4f}%")
    print(f"\n  Mann-Whitney U (one-sided, greedy > random):")
    print(f"    statistic={mw_stat:.4f}  p={mw_pval_one:.4f}  "
          f"({'significant' if mw_pval_one < 0.05 else 'not significant'} at p<0.05)")
    print(f"  t-test (reference):")
    print(f"    statistic={t_stat:.4f}  p={t_pval:.4f}")

    beats_random_mean = best_score > random_mean
    beats_random_max  = best_score > random_max
    print(f"\n  Greedy beats random mean: {beats_random_mean}")
    print(f"  Greedy beats random max:  {beats_random_max}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}\nFULL K SUMMARY\n{'='*65}")
    print(f"  Baseline:    {baseline:.4f}%")
    print(f"  Default CAA: {default_caa:.4f}%\n")
    print(f"  {'K':>4} | {'Greedy':>8} | {'Rand Mean':>10} | "
          f"{'Rand Std':>9} | {'Rand Max':>9} | {'Gap':>8}")
    print(f"  {'─'*4}-+-{'─'*8}-+-{'─'*10}-+-{'─'*9}-+-{'─'*9}-+-{'─'*8}")

    for k in range(1, n_heads + 1):
        r = random_k_results[k]
        # Greedy score at this K: best score found by greedy using exactly K heads
        # Find in greedy all_runs history — the best score at subset size k
        greedy_at_k = None
        for run in greedy.get("forward", {}).get("all_runs", []):
            for h in run.get("history", []):
                if len(h["subset"]) == k:
                    if greedy_at_k is None or h["score"] > greedy_at_k:
                        greedy_at_k = h["score"]
        # Also check backward
        for h in greedy.get("backward", {}).get("history", []):
            if len(h["subset"]) == k:
                if greedy_at_k is None or h["score"] > greedy_at_k:
                    greedy_at_k = h["score"]

        greedy_str = f"{greedy_at_k:.4f}%" if greedy_at_k else "  —  "
        marker     = " <-- K*" if k == greedy_k else ""
        print(f"  {k:>4} | {greedy_str:>8} | {r['mean']:>10.4f}% | "
              f"{r['std']:>9.4f}% | {r['max']:>9.4f}% | "
              f"{(greedy_at_k or 0) - r['mean']:>+8.4f}%{marker}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    model_tag = model_name.replace("/", "_").replace(".", "_")
    out_file  = os.path.join(results_dir,
                              f"random_k_layer{layer}_{timestamp}.json")
    output = {
        "timestamp":        datetime.now().isoformat(),
        "config":           config_path,
        "model":            model_name,
        "dataset":          dataset_key,
        "eval_method":      eval_method,
        "layer":            layer,
        "n_heads":          n_heads,
        "mu":               mu,
        "eval_samples":     eval_samples,
        "n_repeats":        n_repeats,
        "greedy_k":         greedy_k,
        "greedy_score":     best_score,
        "greedy_subset":    best_subset,
        "random_k_results": {str(k): v for k, v in random_k_results.items()},
        "comparison_at_k_star": {
            "k":                greedy_k,
            "greedy_score":     best_score,
            "random_mean":      random_mean,
            "random_std":       random_std,
            "random_max":       random_max,
            "gap":              round(best_score - random_mean, 4),
            "beats_random_mean": beats_random_mean,
            "beats_random_max":  beats_random_max,
            "mannwhitney_u":    mw_stat,
            "pval_one_sided":   round(mw_pval_one, 4),
            "pval_two_sided":   round(mw_pval_two, 4),
            "ttest_pval":       round(t_pval, 4),
            "significant_p05":  mw_pval_one < 0.05,
        },
    }
    # Convert numpy types to plain Python for JSON serialisation
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
