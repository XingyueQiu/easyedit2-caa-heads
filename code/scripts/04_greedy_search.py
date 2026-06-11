#!/usr/bin/env python
# coding: utf-8
"""
scripts/04_greedy_search.py
===========================
Greedy head subset search for a given model and target layer.

Runs:
  Part 1  Forward greedy — starts from every single head (n_head runs)
  Part 2  Backward greedy — starts from all heads, removes one at a time
  Part 3  Overall best — better of forward and backward

The best subset found here is used by:
  05_random_k.py   (RQ3 — is greedy better than random?)
  06_mu_sweep.py   (characterise subset across mu values)
  07_per_input.py  (RQ4 — input-dependent specialisation)
  08_held_out.py   (final evaluation on dev.jsonl)

Reads:  results/<model>/<dataset>/layer<N>/baselines_*.json
        (checks head_api_ok before running)
Writes: results/<model>/<dataset>/layer<N>/greedy_<timestamp>.json

Usage:
    cd ~/EasyEdit_clean
    python code/scripts/04_greedy_search.py --layer 6
    python code/scripts/04_greedy_search.py --layer 6 --config code/configs/config_toxicity-gpt2.yaml
    python code/scripts/04_greedy_search.py --layer 6 --eval sentiment
"""

import os, sys, json, random, statistics, argparse, glob
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
    eval_caa,
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
MU                = 2.0
RANDOM_K_REPEATS  = 30


# ============================================================================
# Cache
# ============================================================================

_cache = {}

def flush_cache():
    global _cache
    _cache = {}

def cached_eval(cfg, model_name, heads, mu, pool,
                layer, results_file, dataset_key, eval_method, label=""):
    """Eval with memoisation — same (heads, mu, layer) returns cached score."""
    # For single-sample calls include input text in key to avoid collision
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
# Forward greedy
# ============================================================================

def greedy_forward(cfg, model_name, n_heads, mu, pool,
                   layer, results_file, dataset_key, eval_method, baseline):
    print(f"\n{'='*65}\nFORWARD GREEDY  (layer {layer})\n{'='*65}")

    # Score every single head first
    single_scores = []
    for h in range(n_heads):
        s = cached_eval(cfg, model_name, [h], mu, pool,
                        layer, results_file, dataset_key, eval_method,
                        label=f"Head {h:2d}")
        single_scores.append({"head": h, "score": s,
                               "improvement": s - baseline})

    sorted_singles = sorted(single_scores, key=lambda x: x["score"], reverse=True)
    ranking_str = "  ".join(
        f"H{x['head']}={x['score']:.2f}%" for x in sorted_singles[:5]
    )
    print(f"\n  Single head ranking (top 5): {ranking_str}")

    # Run forward greedy from every head in sequential order (H0 to H{n-1}).
    # Order does not affect correctness — global best is max across all runs.
    # Sequential order makes convergence plots trivially interpretable:
    # run N always corresponds to head N.
    start_order = list(range(n_heads))
    global_best_subset = None
    global_best_score  = float("-inf")
    global_best_run    = 0
    global_best_iter   = 0
    all_runs           = []

    for run_idx, start_head in enumerate(start_order):
        print(f"\n{'─'*50}")
        print(f"Run {run_idx+1}/{n_heads}  start=H{start_head}")

        current       = [start_head]
        current_score = single_scores[start_head]["score"]
        remaining     = [h for h in range(n_heads) if h != start_head]
        run_best_subset = current.copy()
        run_best_score  = current_score
        run_best_iter   = 0
        history = [{"iteration": 0, "subset": current.copy(),
                    "score": current_score, "added_head": start_head}]

        if current_score > global_best_score:
            global_best_subset = current.copy()
            global_best_score  = current_score
            global_best_run    = run_idx + 1
            global_best_iter   = 0

        for iteration in range(1, n_heads):
            if not remaining:
                break
            best_add = None
            best_add_score = float("-inf")
            for h in remaining:
                candidate = sorted(current + [h])
                s = cached_eval(cfg, model_name, candidate, mu, pool,
                                layer, results_file, dataset_key, eval_method)
                if s > best_add_score:
                    best_add_score = s
                    best_add = h

            current = sorted(current + [best_add])
            current_score = best_add_score
            remaining.remove(best_add)

            if current_score > run_best_score:
                run_best_subset = current.copy()
                run_best_score  = current_score
                run_best_iter   = iteration

            if current_score > global_best_score:
                global_best_subset = current.copy()
                global_best_score  = current_score
                global_best_run    = run_idx + 1
                global_best_iter   = iteration
                print(f"  NEW GLOBAL BEST: {current} -> {current_score:.4f}%")

            history.append({"iteration": iteration, "subset": current.copy(),
                             "score": current_score, "added_head": best_add})

        all_runs.append({
            "run_number":       run_idx + 1,
            "start_head":       start_head,
            "best_subset":      run_best_subset,
            "best_score":       run_best_score,
            "best_iteration":   run_best_iter,
            "history":          history,
        })
        print(f"  Run {run_idx+1} best: {run_best_subset} -> {run_best_score:.4f}%")

    print(f"\nForward global best: {global_best_subset} -> {global_best_score:.4f}%")
    print(f"  (Run {global_best_run}, iteration {global_best_iter})")
    return {
        "best_subset":       global_best_subset,
        "best_score":        global_best_score,
        "best_from_run":     global_best_run,
        "best_at_iteration": global_best_iter,
        "single_scores":     single_scores,
        "all_runs":          all_runs,
    }


# ============================================================================
# Backward greedy
# ============================================================================

def greedy_backward(cfg, model_name, n_heads, mu, pool,
                    layer, results_file, dataset_key, eval_method,
                    baseline, full_score):
    print(f"\n{'='*65}\nBACKWARD GREEDY  (layer {layer})\n{'='*65}")
    print(f"Start: all {n_heads} heads -> {full_score:.4f}%")

    current_subset = list(range(n_heads))
    current_score  = full_score
    best_subset    = current_subset.copy()
    best_score     = current_score
    history = [{"iteration": 0, "removed_head": None,
                "subset": current_subset.copy(),
                "num_heads": n_heads, "score": current_score}]

    for iteration in range(1, n_heads):
        if len(current_subset) <= 1:
            break
        best_rem       = None
        best_rem_score = float("-inf")
        for h in current_subset:
            candidate = [x for x in current_subset if x != h]
            s = cached_eval(cfg, model_name, candidate, mu, pool,
                            layer, results_file, dataset_key, eval_method)
            if s > best_rem_score:
                best_rem_score = s
                best_rem = h

        current_subset = [h for h in current_subset if h != best_rem]
        current_score  = best_rem_score

        if current_score > best_score:
            best_subset = current_subset.copy()
            best_score  = current_score
            print(f"  NEW BEST: removed H{best_rem} -> {current_score:.4f}%")

        history.append({
            "iteration":    iteration,
            "removed_head": best_rem,
            "subset":       current_subset.copy(),
            "num_heads":    len(current_subset),
            "score":        current_score,
        })

    print(f"Backward best: {best_subset} -> {best_score:.4f}%  "
          f"(vs all-heads: {best_score-full_score:+.4f}%)")
    return {
        "best_subset":    best_subset,
        "best_score":     best_score,
        "beats_full":     best_score > full_score,
        "improvement_vs_full": best_score - full_score,
        "history":        history,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Greedy head subset search")
    parser.add_argument("--layer",   type=int, required=True)
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
    results_file = os.path.join(results_dir, "greedy_tmp.json")

    print("=" * 65)
    print("GREEDY HEAD SEARCH")
    print(f"Config:   {config_path}")
    print(f"Model:    {model_name}")
    print(f"Dataset:  {dataset_key}")
    print(f"Layer:    {layer}")
    print(f"Eval:     {eval_method}")
    print(f"MU:       {mu}")
    print(f"Samples:  {eval_samples}")
    print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Pre-flight checks ────────────────────────────────────────────────────
    vec_path = find_vector_file(vec_dir, layer)
    if not vec_path:
        print(f"\n  ERROR: no vector for layer {layer}.")
        print(f"  Run: python code/scripts/00_train_vectors.py --layers {layer}")
        return

    # Load baseline from 03_baselines.py output
    baseline_files = sorted(glob.glob(
        os.path.join(results_dir, "baselines_layer*.json")
    ))
    if not baseline_files:
        print(f"\n  ERROR: no baselines file found in {results_dir}")
        print(f"  Run: python code/scripts/03_baselines.py --layer {layer}")
        return

    with open(baseline_files[-1]) as f:
        bl = json.load(f)

    baseline    = bl["scores"]["no_steering"]
    default_caa = bl["scores"]["default_caa"]
    all_heads_attn = bl["scores"].get("all_heads_explicit", None)

    print(f"\n  Loaded baselines from: {baseline_files[-1]}")
    print(f"  No steering:   {baseline:.4f}%")
    print(f"  Default CAA:   {default_caa:.4f}%")
    if all_heads_attn:
        print(f"  All heads attn: {all_heads_attn:.4f}%")

    # ── Data split ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}\nDATA SPLIT\n{'='*65}")
    _, eval_pool = make_splits(
        cfg, dataset_key=dataset_key,
        train_frac=TRAIN_VECTOR_FRAC,
        eval_cap=eval_samples,
        seed=SPLIT_SEED,
    )

    flush_cache()

    # ── Part 1: Forward greedy ───────────────────────────────────────────────
    forward = greedy_forward(
        cfg, model_name, n_heads, mu, eval_pool,
        layer, results_file, dataset_key, eval_method, baseline
    )

    # ── Part 2: Backward greedy ──────────────────────────────────────────────
    # Backward needs the all-heads attention-level score as its starting point.
    # We explicitly evaluate it here to guarantee the correct injection point
    # (attention-level) regardless of cache state — never fall back to
    # default_caa (block-level) which would make forward and backward
    # incomparable.
    full_heads_key = (tuple(range(n_heads)), mu, layer, None)
    if full_heads_key in _cache:
        full_score = _cache[full_heads_key]
        print(f"\n  All-heads score (from cache): {full_score:.4f}%")
    else:
        print(f"\n{'='*65}")
        print(f"ALL-HEADS BASELINE (backward greedy start point)")
        print(f"{'='*65}")
        set_apply_yaml(APPLY_YAML, GENERATE_YAML, layer, mu,
                       use_head_steering=True,
                       attention_heads=list(range(n_heads)))
        full_score = eval_caa(
            cfg=cfg, pool=eval_pool,
            results_file=results_file,
            format_pool_fn=lambda p: format_pool(
                p, dataset_key=dataset_key, use_negative_prompt=True
            ),
            dataset_key=dataset_key,
            model_name=model_name,
            eval_method=eval_method,
        )
        _cache[full_heads_key] = full_score
        print(f"  All heads (attn-level): {full_score:.4f}%")

    backward = greedy_backward(
        cfg, model_name, n_heads, mu, eval_pool,
        layer, results_file, dataset_key, eval_method,
        baseline, full_score
    )

    # ── Part 3: Overall best ─────────────────────────────────────────────────
    if backward["best_score"] > forward["best_score"]:
        best_subset = backward["best_subset"]
        best_score  = backward["best_score"]
        best_method = "backward"
    else:
        best_subset = forward["best_subset"]
        best_score  = forward["best_score"]
        best_method = "forward"

    print(f"\n{'='*65}\nRESULTS  (layer {layer})\n{'='*65}")
    print(f"  No steering:   {baseline:.4f}%")
    print(f"  Default CAA:   {default_caa:.4f}%  ({default_caa-baseline:+.4f}%)")
    if all_heads_attn:
        print(f"  All heads attn: {all_heads_attn:.4f}%  "
              f"({all_heads_attn-baseline:+.4f}%)")
    print(f"  Greedy best:   {best_score:.4f}%  ({best_score-baseline:+.4f}%)")
    print(f"  Best subset:   {best_subset}  (K={len(best_subset)}, "
          f"found by {best_method} greedy)")
    print(f"\n  vs default CAA: {best_score-default_caa:+.4f}%  "
          f"({'BETTER' if best_score > default_caa else 'WORSE'})")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_file = os.path.join(results_dir,
                             f"greedy_layer{layer}_{timestamp}.json")
    output = {
        "timestamp":       datetime.now().isoformat(),
        "config":          config_path,
        "model":           model_name,
        "dataset":         dataset_key,
        "eval_method":     eval_method,
        "layer":           layer,
        "n_heads":         n_heads,
        "mu":              mu,
        "eval_samples":    eval_samples,
        "baselines": {
            "no_steering":       baseline,
            "default_caa":       default_caa,
            "all_heads_attn":    all_heads_attn,
        },
        "forward":         forward,
        "backward":        backward,
        "best_subset":     best_subset,
        "best_score":      best_score,
        "best_method":     best_method,
        "best_k":          len(best_subset),
        "vs_default_caa":  round(best_score - default_caa, 4),
        "vs_baseline":     round(best_score - baseline, 4),
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
    print(f"\n  Next steps:")
    print(f"    05_random_k.py    --layer {layer}  (RQ3)")
    print(f"    06_mu_sweep.py    --layer {layer}  (characterise subset)")
    print(f"    07_per_input.py   --layer {layer}  (RQ4)")
    print(f"    08_held_out.py    --layer {layer}  (final eval)")

    if os.path.exists(results_file):
        os.remove(results_file)

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
