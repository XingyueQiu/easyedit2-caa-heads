#!/usr/bin/env python
# coding: utf-8
"""
scripts/07_per_input.py
=======================
Per-input head specialisation analysis for RQ2 (revised):
  "Which attention heads are most important for steering, and is this
   consistent across inputs or input-dependent?"

For each of N_INPUTS inputs from the val pool, evaluates:
  K=1: all 12 single heads — finds best head per input
  K=2: all C(12,2)=66 pairs — finds best pair per input (exhaustive)

Produces:
  - K=1 head frequency histogram (which head is best most often)
  - K=2 pair frequency histogram (which pair is best most often)
  - Score curves: mean score at K=1 and K=2 vs global greedy subset
  - Chi-squared test: is K=1 head frequency non-uniform? (primary test)
  - Wilcoxon test: does per-input K=2 best pair beat global greedy subset?

Fixed K=1,2 regardless of greedy K* — directly answers which heads
carry the steering signal rather than optimising subset size.

Reads:  results/<model>/<dataset>/layer<N>/greedy_*.json
Writes: results/<model>/<dataset>/layer<N>/per_input_<timestamp>.json

Usage:
    cd ~/EasyEdit_clean
    python code/scripts/07_per_input.py --layer 8
    python code/scripts/07_per_input.py --layer 8 --n-inputs 100
    python code/scripts/07_per_input.py --layer 8 --eval sentiment
"""

import os, sys, json, random, statistics, argparse, glob
from itertools import combinations
from collections import Counter
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon as scipy_wilcoxon
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
    eval_caa,
)

# ============================================================================
# Defaults
# ============================================================================

DEFAULT_CONFIG    = os.path.join(_CODE_DIR, "configs", "config_toxicity-gpt2.yaml")
GENERATE_YAML     = os.path.join(_PROJECT_DIR, "hparams", "Steer", "caa_hparams", "generate_caa.yaml")
APPLY_YAML        = os.path.join(_PROJECT_DIR, "hparams", "Steer", "caa_hparams", "apply_caa.yaml")

SPLIT_SEED        = 42
TRAIN_VECTOR_FRAC = 0.80
EVAL_NUM_SAMPLES  = 200
PER_INPUT_SAMPLES = 100
MU                = 2.0


# ============================================================================
# Per-sample cache
# ============================================================================

_cache = {}

def cached_eval_single(cfg, model_name, heads, mu, sample,
                        layer, results_dir, dataset_key, eval_method):
    input_id = (sample.get("not_matching") or sample.get("input", ""))[:80]
    key = (tuple(sorted(heads)), mu, layer, input_id)
    if key in _cache:
        return _cache[key]
    set_apply_yaml(APPLY_YAML, GENERATE_YAML, layer, mu,
                   use_head_steering=True, attention_heads=list(heads))
    results_file = os.path.join(results_dir, "per_input_tmp.json")
    score = eval_caa(
        cfg=cfg, pool=[sample],
        results_file=results_file,
        format_pool_fn=lambda p: format_pool(
            p, dataset_key=dataset_key, use_negative_prompt=True
        ),
        dataset_key=dataset_key,
        model_name=model_name,
        eval_method=eval_method,
    )
    _cache[key] = score
    return score


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Per-input head specialisation analysis")
    parser.add_argument("--layer",    type=int, required=True)
    parser.add_argument("--config",   default=DEFAULT_CONFIG)
    parser.add_argument("--eval",     default="toxigen",
                        choices=["toxigen", "sentiment"])
    parser.add_argument("--mu",       type=float, default=MU)
    parser.add_argument("--n-inputs", type=int,   default=PER_INPUT_SAMPLES)
    parser.add_argument("--samples",  type=int,   default=EVAL_NUM_SAMPLES)
    args = parser.parse_args()

    layer        = args.layer
    config_path  = args.config
    eval_method  = args.eval
    mu           = args.mu
    n_inputs     = args.n_inputs
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

    from math import comb
    n_pairs = comb(n_heads, 2)

    print("=" * 65)
    print("PER-INPUT ANALYSIS  (K=1 and K=2)")
    print(f"Config:    {config_path}")
    print(f"Model:     {model_name}")
    print(f"Dataset:   {dataset_key}")
    print(f"Layer:     {layer}")
    print(f"Eval:      {eval_method}")
    print(f"MU:        {mu}")
    print(f"N inputs:  {n_inputs}")
    print(f"K=1: {n_heads} single heads per input")
    print(f"K=2: {n_pairs} pairs per input (exhaustive, C({n_heads},2))")
    print(f"Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Pre-flight ───────────────────────────────────────────────────────────
    if not find_vector_file(vec_dir, layer):
        print(f"\n  ERROR: no vector for layer {layer}.")
        print(f"  Run: python code/scripts/00_train_vectors.py --layers {layer}")
        return

    # Load greedy results for global subset comparison
    greedy_files = sorted(glob.glob(
        os.path.join(results_dir, "greedy_layer*.json")))
    global_subset = None
    global_k      = None
    if greedy_files:
        with open(greedy_files[-1]) as f:
            greedy = json.load(f)
        global_subset = greedy["best_subset"]
        global_k      = len(global_subset)
        print(f"\n  Global greedy subset: {global_subset}  K={global_k}")
    else:
        print(f"\n  No greedy results found — skipping global subset comparison.")

    # ── Data ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}\nDATA\n{'='*65}")
    _, eval_pool = make_splits(
        cfg, dataset_key=dataset_key,
        train_frac=TRAIN_VECTOR_FRAC,
        eval_cap=eval_samples,
        seed=SPLIT_SEED,
    )
    rng     = random.Random(SPLIT_SEED + 777)
    samples = rng.sample(eval_pool, min(n_inputs, len(eval_pool)))
    print(f"  Val pool: {len(eval_pool)}  Sampled: {len(samples)}")

    # ── Per-input analysis ────────────────────────────────────────────────────
    print(f"\n{'='*65}\nPER-INPUT ANALYSIS\n{'='*65}")

    # Frequency counters
    head_freq_k1  = Counter()            # best single head per input
    pair_freq_k2  = Counter()            # best pair per input

    per_input_results = []

    for i, sample in enumerate(samples):
        text = (sample.get("not_matching") or
                sample.get("input") or "")[:60]
        print(f"\n  [{i+1}/{len(samples)}] {text}...")

        # ── K=1: score all single heads ──────────────────────────────────────
        k1_scores = {}
        for h in range(n_heads):
            s = cached_eval_single(cfg, model_name, [h], mu,
                                    sample, layer, results_dir,
                                    dataset_key, eval_method)
            k1_scores[h] = s

        best_head_k1  = max(k1_scores, key=k1_scores.get)
        best_score_k1 = k1_scores[best_head_k1]
        head_freq_k1[best_head_k1] += 1

        # ── K=2: exhaustive over all C(n,2) pairs ────────────────────────────
        k2_scores = {}
        for pair in combinations(range(n_heads), 2):
            s = cached_eval_single(cfg, model_name, list(pair), mu,
                                    sample, layer, results_dir,
                                    dataset_key, eval_method)
            k2_scores[pair] = s

        best_pair_k2  = max(k2_scores, key=k2_scores.get)
        best_score_k2 = k2_scores[best_pair_k2]
        pair_freq_k2[best_pair_k2] += 1

        # ── Compare against global greedy subset ─────────────────────────────
        global_score_here = None
        if global_subset:
            global_score_here = cached_eval_single(
                cfg, model_name, global_subset, mu,
                sample, layer, results_dir, dataset_key, eval_method
            )

        print(f"    K=1 best: H{best_head_k1}={best_score_k1:.2f}%  "
              f"K=2 best: {list(best_pair_k2)}={best_score_k2:.2f}%"
              + (f"  Global(K={global_k})={global_score_here:.2f}%"
                 if global_score_here else ""))

        per_input_results.append({
            "input_idx":           i,
            "text":                text,
            "k1_best_head":        best_head_k1,
            "k1_best_score":       round(best_score_k1, 4),
            "k1_scores":           {str(h): round(s, 4)
                                     for h, s in k1_scores.items()},
            "k2_best_pair":        list(best_pair_k2),
            "k2_best_score":       round(best_score_k2, 4),
            "global_subset_score": round(global_score_here, 4)
                                   if global_score_here else None,
            "k2_vs_global":        round(best_score_k2 - global_score_here, 4)
                                   if global_score_here else None,
        })

    # ── Statistical analysis ──────────────────────────────────────────────────
    print(f"\n{'='*65}\nSTATISTICAL ANALYSIS\n{'='*65}")

    # ── K=1 frequency histogram ───────────────────────────────────────────────
    print(f"\n  K=1 Head frequency (best single head per input):")
    for h in range(n_heads):
        count = head_freq_k1.get(h, 0)
        bar   = "█" * count
        print(f"    H{h:2d}: {count:3d}  {bar}")

    top3_heads = head_freq_k1.most_common(3)
    print(f"\n  Top-3 heads: {[(f'H{h}', c) for h, c in top3_heads]}")
    print(f"  Unique heads chosen: "
          f"{len(head_freq_k1)}/{n_heads}")

    # Chi-squared: is K=1 distribution non-uniform?
    observed_k1 = [head_freq_k1.get(h, 0) for h in range(n_heads)]
    expected_k1 = [len(samples) / n_heads] * n_heads
    chi2_stat, chi2_pval = stats.chisquare(observed_k1, f_exp=expected_k1)
    print(f"\n  Chi-squared (K=1 head frequency vs uniform):")
    print(f"    chi2={chi2_stat:.4f}  p={chi2_pval:.4f}  df={n_heads-1}")
    print(f"    {'Non-uniform' if chi2_pval < 0.05 else 'Uniform'} "
          f"at p<0.05 — head importance is "
          f"{'structured' if chi2_pval < 0.05 else 'not significantly structured'}")

    # ── K=2 frequency histogram ───────────────────────────────────────────────
    print(f"\n  K=2 Pair frequency (top 10 most chosen pairs):")
    for pair, count in pair_freq_k2.most_common(10):
        bar = "█" * count
        print(f"    {list(pair)}: {count:3d}  {bar}")
    print(f"  Unique pairs chosen: {len(pair_freq_k2)}/{n_pairs}")

    # ── K=2 vs global greedy subset ───────────────────────────────────────────
    if global_subset:
        k2_vs_global = [r["k2_vs_global"] for r in per_input_results
                        if r["k2_vs_global"] is not None]
        print(f"\n  K=2 per-input best vs global greedy subset (K={global_k}):")
        print(f"    Mean gain: {np.mean(k2_vs_global):+.4f}%")
        print(f"    Std:       {np.std(k2_vs_global):.4f}%")
        print(f"    Wins (K=2 > global): "
              f"{sum(g > 0 for g in k2_vs_global)}/{len(k2_vs_global)}")

        # Wilcoxon: does per-input K=2 beat global subset?
        nonzero = [g for g in k2_vs_global if g != 0.0]
        if len(nonzero) >= 10:
            w_stat, w_pval = scipy_wilcoxon(
                k2_vs_global, alternative="greater", zero_method="wilcox"
            )
            print(f"\n  Wilcoxon (per-input K=2 gain > 0):")
            print(f"    stat={w_stat:.4f}  p={w_pval:.4f}  "
                  f"{'Significant' if w_pval < 0.05 else 'Not significant'} at p<0.05")
        else:
            w_stat, w_pval = None, None
            print(f"\n  Wilcoxon skipped (too few non-zero gains)")

    # ── Score summary ─────────────────────────────────────────────────────────
    k1_scores_all = [r["k1_best_score"] for r in per_input_results]
    k2_scores_all = [r["k2_best_score"] for r in per_input_results]
    print(f"\n  Mean scores across {len(samples)} inputs:")
    print(f"    K=1 best head mean: {np.mean(k1_scores_all):.4f}%")
    print(f"    K=2 best pair mean: {np.mean(k2_scores_all):.4f}%")
    if global_subset and k2_vs_global:
        global_scores = [r["global_subset_score"] for r in per_input_results
                         if r["global_subset_score"] is not None]
        print(f"    Global K={global_k} mean:   {np.mean(global_scores):.4f}%")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_file = os.path.join(results_dir,
                             f"per_input_layer{layer}_{timestamp}.json")

    
    def _json_safe(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Not JSON serialisable: {type(obj)}")

    output = {
        "timestamp":         datetime.now().isoformat(),
        "config":            config_path,
        "model":             model_name,
        "dataset":           dataset_key,
        "eval_method":       eval_method,
        "layer":             layer,
        "n_heads":           n_heads,
        "mu":                mu,
        "n_inputs":          len(samples),
        "global_subset":     global_subset,
        "global_k":          global_k,
        "per_input_results": per_input_results,
        "k1_head_frequency": {str(h): head_freq_k1.get(h, 0)
                               for h in range(n_heads)},
        "k2_pair_frequency": {str(list(p)): c
                               for p, c in pair_freq_k2.most_common()},
        "k1_top3_heads":     [(h, c) for h, c in top3_heads],
        "k1_unique_chosen":  len(head_freq_k1),
        "k2_unique_chosen":  len(pair_freq_k2),
        "chisquare_k1": {
            "stat":            round(float(chi2_stat), 4),
            "pval":            round(float(chi2_pval), 4),
            "df":              n_heads - 1,
            "significant_p05": bool(chi2_pval < 0.05),
            "observed":        observed_k1,
            "expected":        [round(e, 2) for e in expected_k1],
        },
        "score_summary": {
            "k1_mean": round(float(np.mean(k1_scores_all)), 4),
            "k2_mean": round(float(np.mean(k2_scores_all)), 4),
        },
    }

    if global_subset and k2_vs_global:
        output["k2_vs_global"] = {
            "mean_gain": round(float(np.mean(k2_vs_global)), 4),
            "std":       round(float(np.std(k2_vs_global)), 4),
            "wins":      int(sum(g > 0 for g in k2_vs_global)),
            "n":         len(k2_vs_global),
            "wilcoxon":  {
                "stat": float(w_stat) if w_stat else None,
                "pval": round(float(w_pval), 4) if w_pval else None,
                "significant_p05": bool(w_pval < 0.05) if w_pval else None,
            } if w_stat else None,
        }

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=_json_safe)
    print(f"\n  Results saved: {out_file}")

    tmp = os.path.join(results_dir, "per_input_tmp.json")
    if os.path.exists(tmp):
        os.remove(tmp)

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
