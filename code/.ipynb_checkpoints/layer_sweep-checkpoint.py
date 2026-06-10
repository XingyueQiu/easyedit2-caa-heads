#!/usr/bin/env python
# coding: utf-8
"""
layer_sweep.py
==============
Full layer sweep for layer selection.

For each layer (0 to n_layers-1):
  1. Train steering vector if layer_N.pt does not exist
  2. Evaluate baseline (once, reused across layers)
  3. Evaluate default CAA
  4. Record score

Selects the layer with highest default CAA score as the optimal
intervention layer for the full greedy head selection pipeline.

Usage:
    python layer_sweep.py
    python layer_sweep.py --model qwen   # switch model

Results saved to layer_sweep_MODELNAME_TIMESTAMP.json
"""

import os, sys, json, random, subprocess, re, yaml
import torch
from datetime import datetime

sys.path.append("../")
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from steer.datasets import prepare_train_dataset
from steer.vector_generators.vector_generators import BaseVectorGenerator
from steer.vector_appliers.vector_applier import BaseVectorApplier

# ============================================================================
# Configuration  <- edit these
# ============================================================================

CONFIG_PATH       = "./config_toxicity-gpt2.yaml"
# CONFIG_PATH     = "./config_toxicity-qwen1.5-1.8b.yaml"
# CONFIG_PATH     = "./config_toxicity-qwen2.5-0.5b.yaml"
#CONFIG_PATH       = "./config_sst2pair-gpt2.yaml"

SPLIT_SEED        = 42
TRAIN_VECTOR_FRAC = 0.80
EVAL_NUM_SAMPLES  = 200
USE_TOXIC_PROMPTS = True
GENERATION_SEED   = 42
MU                = 2.0

GENERATE_YAML = "../hparams/Steer/caa_hparams/generate_caa.yaml"
APPLY_YAML    = "../hparams/Steer/caa_hparams/apply_caa.yaml"
RESULTS_FILE  = "vectors/layer_sweep_tmp.json"

# ============================================================================
# Helpers
# ============================================================================

def first_path(v):
    if v is None: return None
    if isinstance(v, str): return v
    return str(v[0]) if len(v) > 0 else None

def load_config():
    cfg      = OmegaConf.load(CONFIG_PATH)
    out_dir  = first_path(cfg.get("steer_vector_output_dirs"))
    load_dir = first_path(cfg.get("steer_vector_load_dir"))
    if not load_dir: load_dir = out_dir
    if not out_dir:  out_dir  = load_dir
    cfg.steer_vector_output_dirs = [out_dir]
    cfg.steer_vector_load_dir    = [load_dir]
    return cfg

def load_config_raw():
    """Raw config for BaseVectorGenerator — no modification."""
    return OmegaConf.load(CONFIG_PATH)

def get_n_layers(model_name):
    c = AutoConfig.from_pretrained(model_name)
    return getattr(c, "num_hidden_layers", getattr(c, "n_layer", 12))

def get_n_head(model_name):
    c = AutoConfig.from_pretrained(model_name)
    return getattr(c, "num_attention_heads", getattr(c, "n_head", 12))

def extract_toxigen_score(text):
    for pat in [r"toxigen_overall':\s*([0-9]+\.[0-9]+|[0-9]+)",
                r"toxigen_overall[^0-9]*([0-9]+\.[0-9]+|[0-9]+)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            v = float(m.group(1))
            return v * 100.0 if v <= 1.0 else v
    return None

def run_toxigen_eval(results_path, model_name):
    cmd = ["python", "../steer/evaluate/evaluate.py",
           "--generation_dataset_path", results_path,
           "--eval_methods", "toxigen",
           "--model_name_or_path", model_name]
    p   = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"Toxigen eval failed:\n{out[:800]}")
    score = extract_toxigen_score(out)
    if score is None:
        raise RuntimeError(f"Could not parse score:\n{out[:500]}")
    return score

def set_generate_yaml_layer(layer):
    """Patch generate_caa.yaml to target a specific layer."""
    with open(GENERATE_YAML) as f: g = yaml.safe_load(f)
    g["layers"] = [layer]
    with open(GENERATE_YAML, "w") as f: yaml.dump(g, f)

def set_apply_yaml(layer, mu, use_head_steering=False):
    """Patch apply_caa.yaml for evaluation."""
    with open(APPLY_YAML) as f: a = yaml.safe_load(f)
    with open(GENERATE_YAML) as f: g = yaml.safe_load(f)
    a["layers"]            = [layer]
    g["layers"]            = [layer]
    a["use_head_steering"] = use_head_steering
    g["use_head_steering"] = use_head_steering
    a["attention_heads"]   = []
    g["attention_heads"]   = []
    a["multipliers"]       = [float(mu)]
    with open(APPLY_YAML, "w") as f: yaml.dump(a, f)
    with open(GENERATE_YAML, "w") as f: yaml.dump(g, f)

def format_pool(pool):
    out = []
    for ex in pool:
        text = (ex.get("not_matching") if USE_TOXIC_PROMPTS
                else ex.get("matching")) \
               or ex.get("input") or ex.get("prompt")
        if text:
            out.append({
                "matching":           text,
                "input":              text,
                "reference_response": ex.get(
                    "matching" if USE_TOXIC_PROMPTS else "not_matching", ""),
            })
    return {"toxicity": out}

# ============================================================================
# Data split
# ============================================================================

def make_splits(cfg):
    datasets  = prepare_train_dataset(cfg)
    data_list = datasets["toxicity"]
    n         = len(data_list)
    rng       = random.Random(SPLIT_SEED)
    idx       = list(range(n))
    rng.shuffle(idx)
    train_end  = int(TRAIN_VECTOR_FRAC * n)
    val_idx    = idx[train_end:]
    train_data = [data_list[i] for i in idx[:train_end]]
    eval_pool  = [data_list[i] for i in val_idx][:EVAL_NUM_SAMPLES]
    print(f"  Total: {n}  Train: {len(train_data)}  "
          f"Eval pool: {len(eval_pool)}")
    return train_data, eval_pool

# ============================================================================
# Vector training
# ============================================================================

def train_vector_for_layer(layer, train_data):
    """
    Train CAA vector for a specific layer.
    Uses raw config (identical to train_vectors.py).
    Patches generate_caa.yaml layer field before training.
    """
    print(f"  Training vector for layer {layer} ...")
    set_generate_yaml_layer(layer)
    cfg_raw = load_config_raw()
    generator = BaseVectorGenerator(cfg_raw)
    generator.generate_vectors({"toxicity": train_data})
    print(f"  Vector training complete for layer {layer}.")

def find_vector_file(vec_dir, layer):
    """Find the actual saved vector file for a layer."""
    # EasyEdit saves to vec_dir/toxicity/caa_vector/layer_N.pt
    candidates = [
        os.path.join(vec_dir, f"layer_{layer}.pt"),
        os.path.join(vec_dir, "toxicity", "caa_vector", f"layer_{layer}.pt"),
        os.path.join(vec_dir, "caa_vector", f"layer_{layer}.pt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    # Recursive search as fallback
    for root, dirs, files in os.walk(vec_dir):
        for f in files:
            if f == f"layer_{layer}.pt":
                return os.path.join(root, f)
    return None

def ensure_vector(vec_dir, layer, train_data):
    """Train vector for layer if not already present. Returns path to vector."""
    path = find_vector_file(vec_dir, layer)
    if path:
        vec = torch.load(path, map_location="cpu")
        print(f"  layer_{layer}.pt found: {path}  "
              f"shape={vec.shape}  norm={vec.norm().item():.4f}")
        return path
    else:
        print(f"  layer_{layer}.pt not found — training now ...")
        train_vector_for_layer(layer, train_data)
        path = find_vector_file(vec_dir, layer)
        if not path:
            raise FileNotFoundError(
                f"Vector not found after training for layer {layer}. "
                f"Check EasyEdit save path logic.")
        vec = torch.load(path, map_location="cpu")
        print(f"  Trained and saved: {path}  "
              f"shape={vec.shape}  norm={vec.norm().item():.4f}")
        return path

# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def eval_baseline(cfg, model_name, pool):
    """Evaluate baseline once. Returns score."""
    device = cfg.get("device", "cuda:0")
    tok    = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model  = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    subset = format_pool(pool)["toxicity"]
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    records = []
    for i, ex in enumerate(subset):
        inp = tok(ex["input"], return_tensors="pt", truncation=True,
                  max_length=tok.model_max_length).to(device)
        if GENERATION_SEED is not None:
            torch.manual_seed(GENERATION_SEED + i)
        gen = model.generate(**inp, max_new_tokens=64, do_sample=False)
        records.append({
            "input":              ex["input"],
            "pred":               [tok.decode(gen[0], skip_special_tokens=True)],
            "reference_response": ex.get("reference_response", ""),
        })
    with open(RESULTS_FILE, "w") as f: json.dump(records, f)
    del model; torch.cuda.empty_cache()
    return run_toxigen_eval(RESULTS_FILE, model_name)

def eval_default_caa(cfg, model_name, pool, layer):
    """Evaluate default CAA (block-level) at a specific layer."""
    set_apply_yaml(layer, MU, use_head_steering=False)
    applier = BaseVectorApplier(cfg)
    applier.apply_vectors()
    if getattr(applier, "model", None) is not None:
        applier.model.set_from_positions(0)
    formatted = applier.generate(format_pool(pool), save_results=False)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f: json.dump(formatted, f)
    if getattr(applier, "model", None) is not None:
        applier.model.reset_all()
    del applier; torch.cuda.empty_cache()
    return run_toxigen_eval(RESULTS_FILE, model_name)

def eval_single_head(cfg, model_name, pool, layer, head):
    with open(APPLY_YAML) as f: a = yaml.safe_load(f)
    with open(GENERATE_YAML) as f: g = yaml.safe_load(f)
    a["layers"]            = [layer]
    g["layers"]            = [layer]
    a["use_head_steering"] = True
    g["use_head_steering"] = True
    a["attention_heads"]   = [head]
    g["attention_heads"]   = [head]
    a["multipliers"]       = [float(MU)]
    with open(APPLY_YAML, "w") as f: yaml.dump(a, f)
    with open(GENERATE_YAML, "w") as f: yaml.dump(g, f)
    applier = BaseVectorApplier(cfg)
    applier.apply_vectors()
    if getattr(applier, "model", None) is not None:
        applier.model.set_from_positions(0)
    formatted = applier.generate(format_pool(pool), save_results=False)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f: json.dump(formatted, f)
    if getattr(applier, "model", None) is not None:
        applier.model.reset_all()
    del applier; torch.cuda.empty_cache()
    return run_toxigen_eval(RESULTS_FILE, model_name)

# ============================================================================
# Main
# ============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 65)
    print("FULL LAYER SWEEP — default CAA layer sweep")
    print(f"Config:  {CONFIG_PATH}")
    print(f"MU:      {MU}")
    print(f"Samples: {EVAL_NUM_SAMPLES}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # Load configs
    cfg        = load_config()
    model_name = cfg.get("model_name_or_path", "gpt2")
    vec_dir    = first_path(cfg.get("steer_vector_load_dir"))
    n_layers   = get_n_layers(model_name)
    n_head     = get_n_head(model_name)

    print(f"\nModel:    {model_name}")
    print(f"Layers:   {n_layers}")
    print(f"Heads:    {n_head}")
    print(f"Vec dir:  {vec_dir}")

    # Data split
    print(f"\n{'='*65}")
    print("DATA SPLIT")
    print(f"{'='*65}")
    train_data, eval_pool = make_splits(cfg)

    # Baseline — evaluated once, reused across all layers
    print(f"\n{'='*65}")
    print("BASELINE (no steering)")
    print(f"{'='*65}")
    baseline = eval_baseline(cfg, model_name, eval_pool)
    print(f"  Baseline: {baseline:.4f}%")

    # Layer sweep
    print(f"\n{'='*65}")
    print(f"SWEEPING ALL {n_layers} LAYERS")
    print(f"{'='*65}")

    layer_results = {}

    for layer in range(n_layers):
        print(f"\n{'─'*65}")
        print(f"Layer {layer}/{n_layers-1}")
        print(f"{'─'*65}")

        start_time = datetime.now()

        # Step 1: ensure vector exists
        try:
            vec_path = ensure_vector(vec_dir, layer, train_data)
        except Exception as e:
            print(f"  ERROR training/finding vector for layer {layer}: {e}")
            layer_results[layer] = {
                "default_caa":  None,
                "baseline":     float(baseline),
                "improvement":  None,
                "error":        str(e),
                "duration_min": None,
            }
            continue

        # Step 2: evaluate default CAA
        try:
            score = eval_default_caa(cfg, model_name, eval_pool, layer)
            print(f"  Default CAA score: {score:.4f}%")
            improvement = score - baseline
            duration = (datetime.now() - start_time).total_seconds() / 60

            import statistics as _stats
            alpha = 0.5
            single_scores   = [eval_single_head(cfg, model_name, eval_pool, layer, h)
                            for h in range(n_head)]
            print(f"  Single head scores: {[round(s, 2) for s in single_scores]}")
            dominance       = max(single_scores) - _stats.mean(single_scores)
            selection_score = score - alpha * dominance

            layer_results[layer] = {
                "default_caa":  float(score),
                "baseline":     float(baseline),
                "improvement":  float(improvement),
                "vec_path":     vec_path,
                "duration_min": round(duration, 2),
                "single_head_scores": single_scores,
                "single_head_max":    round(max(single_scores), 4),
                "single_head_mean":   round(_stats.mean(single_scores), 4),
                "dominance":          round(dominance, 4),
                "selection_score":    round(selection_score, 4),
            }
            print(f"  Layer {layer}: {score:.4f}%  "
                  f"(vs baseline: {improvement:+.4f}%)  "
                  f"[{duration:.1f} min]")
            print(f"  Dominance: {dominance:.4f}%  "
                  f"Selection score: {selection_score:.4f}%")

        except Exception as e:
            print(f"  ERROR evaluating layer {layer}: {e}")
            layer_results[layer] = {
                "default_caa":  None,
                "baseline":     float(baseline),
                "improvement":  None,
                "vec_path":     vec_path,
                "error":        str(e),
                "duration_min": None,
            }

    # Select best layer (highest default CAA score, excluding errors)
    valid = {l: r for l, r in layer_results.items()
             if r["default_caa"] is not None}

    if not valid:
        print("\nERROR: No layers evaluated successfully.")
        return

    best_layer = max(valid, key=lambda l: valid[l]["selection_score"])
    best_score = valid[best_layer]["default_caa"]

    # Summary table
    print(f"\n{'='*65}")
    print("LAYER SWEEP SUMMARY")
    print(f"{'='*65}")
    print(f"  Baseline: {baseline:.4f}%\n")
    print(f"  {'Layer':>6} | {'Default CAA':>12} | {'vs Baseline':>12} | {'Dominance':>10} | {'Sel Score':>10} | {'Note':>8}")
    print(f"  {'─'*6}-+-{'─'*12}-+-{'─'*12}-+-{'─'*10}-+-{'─'*10}-+-{'─'*8}")

    for layer in range(n_layers):
        r = layer_results.get(layer, {})
        if r.get("default_caa") is None:
            print(f"  {layer:>6} | {'ERROR':>12} | {'─':>12} |")
            continue
        selected = "<-- BEST" if layer == best_layer else ""
        print(f"  {layer:>6} | {r['default_caa']:>12.4f}% | "
              f"{r['improvement']:>+12.4f}% | "
              f"{r.get('dominance', 0):>10.4f}% | "
              f"{r.get('selection_score', 0):>10.4f}% | {selected}")

    print(f"\n  Best layer: {best_layer}  "
          f"(default CAA: {best_score:.4f}%  "
          f"dominance: {valid[best_layer]['dominance']:.4f}%  "
          f"selection score: {valid[best_layer]['selection_score']:.4f}%)")
    print(f"\n  Next step: set TARGET_LAYERS = [{best_layer}] in run_gpt2.py "
          f"/ run_qwen.py and launch the full experiment.")

    # Save results
    model_tag  = model_name.replace("/", "_").replace(".", "_")
    out_file   = f"layer_sweep_{model_tag}_{timestamp}.json"
    output = {
        "timestamp":      datetime.now().isoformat(),
        "config":         CONFIG_PATH,
        "model":          model_name,
        "n_layers":       n_layers,
        "n_head":         n_head,
        "mu":             MU,
        "eval_samples":   EVAL_NUM_SAMPLES,
        "baseline":       float(baseline),
        "layer_results":  {str(l): r for l, r in layer_results.items()},
        "best_layer":     best_layer,
        "best_score":     float(best_score),
        "recommendation": f"Use layer {best_layer} for full greedy pipeline.",
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {out_file}")

    # Clean up temp file
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
