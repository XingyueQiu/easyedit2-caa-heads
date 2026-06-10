"""
utils/eval.py
=============
Evaluation runners shared across all experiment scripts.

Supports:
  - Toxicity evaluation via tomh/toxigen_roberta
  - Sentiment evaluation via HuggingFace sentiment pipeline (for SST2)

Each function takes a results JSON path, runs the evaluator as a subprocess
(matching the pattern in layer_sweep.py), and returns a float score where
higher is always better:
  - Toxicity:  safety score = 1 - P(toxic), range [0, 100]
  - Sentiment: positive sentiment rate, range [0, 100]
"""

import os
import re
import json
import subprocess
import torch
from typing import List, Dict, Optional

# Path to evaluate.py — resolved relative to this file's location
_UTILS_DIR   = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR    = os.path.dirname(_UTILS_DIR)
_PROJECT_DIR = os.path.dirname(_CODE_DIR)
EVALUATE_PY  = os.path.join(_PROJECT_DIR, "steer", "evaluate", "evaluate.py")


# ── Score extractors ─────────────────────────────────────────────────────────

def extract_toxigen_score(text: str) -> Optional[float]:
    """Parse toxigen_overall score from evaluate.py stdout."""
    for pat in [r"toxigen_overall':\s*([0-9]+\.[0-9]+|[0-9]+)",
                r"toxigen_overall[^0-9]*([0-9]+\.[0-9]+|[0-9]+)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            v = float(m.group(1))
            return v * 100.0 if v <= 1.0 else v
    return None


def extract_sentiment_score(text: str) -> Optional[float]:
    """Parse mean_positive_sentiment score from evaluate.py stdout."""
    for pat in [r"mean_positive_sentiment':\s*([0-9]+\.[0-9]+|[0-9]+)",
                r"mean_positive_sentiment[^0-9]*([0-9]+\.[0-9]+|[0-9]+)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            v = float(m.group(1))
            return v * 100.0 if v <= 1.0 else v
    return None


# ── Subprocess eval runners ──────────────────────────────────────────────────

def run_toxigen_eval(results_path: str, model_name: str) -> float:
    """
    Run toxigen evaluation on a results JSON file.
    Returns safety score in [0, 100] where higher = safer.
    """
    cmd = ["python", EVALUATE_PY,
           "--generation_dataset_path", results_path,
           "--eval_methods", "toxigen",
           "--model_name_or_path", model_name]
    p   = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"Toxigen eval failed:\n{out[:800]}")
    score = extract_toxigen_score(out)
    if score is None:
        raise RuntimeError(f"Could not parse toxigen score:\n{out[:500]}")
    return score


def run_sentiment_eval(results_path: str) -> float:
    """
    Run sentiment evaluation on a results JSON file.
    Returns positive sentiment rate in [0, 100] where higher = more positive.
    """
    cmd = ["python", EVALUATE_PY,
           "--generation_dataset_path", results_path,
           "--eval_methods", "sentiment"]
    p   = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"Sentiment eval failed:\n{out[:800]}")
    score = extract_sentiment_score(out)
    if score is None:
        raise RuntimeError(f"Could not parse sentiment score:\n{out[:500]}")
    return score


def run_eval(results_path: str, model_name: str,
             eval_method: str = "toxigen") -> float:
    """
    Unified eval dispatcher. Returns score in [0, 100], higher = better.

    Args:
        results_path: Path to generation results JSON
        model_name:   Model name/path (required for toxigen)
        eval_method:  "toxigen" or "sentiment"
    """
    if eval_method == "toxigen":
        return run_toxigen_eval(results_path, model_name)
    elif eval_method == "sentiment":
        return run_sentiment_eval(results_path)
    else:
        raise ValueError(f"Unknown eval_method: '{eval_method}'. "
                         "Choose 'toxigen' or 'sentiment'.")


# ── In-process eval functions ─────────────────────────────────────────────────

@torch.no_grad()
def eval_baseline(cfg, model_name: str, pool: List[Dict],
                  results_file: str, format_pool_fn,
                  dataset_key: str = "toxicity",
                  generation_seed: int = 42,
                  eval_method: str = "toxigen") -> float:
    """
    Evaluate baseline (no steering) on pool. Loads and deletes model internally.
    Returns score in [0, 100].
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = cfg.get("device", "cuda:0")
    tok    = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model  = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    subset = format_pool_fn(pool)[dataset_key]
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    records = []
    for i, ex in enumerate(subset):
        inp = tok(ex["input"], return_tensors="pt",
                  truncation=True, max_length=512).to(device)
        if generation_seed is not None:
            torch.manual_seed(generation_seed + i)
        gen = model.generate(**inp, max_new_tokens=64, do_sample=False)
        records.append({
            "input":              ex["input"],
            "pred":               [tok.decode(gen[0], skip_special_tokens=True)],
            "reference_response": ex.get("reference_response", ""),
        })
    with open(results_file, "w") as f:
        json.dump(records, f)
    del model
    torch.cuda.empty_cache()
    return run_eval(results_file, model_name, eval_method)


def eval_caa(cfg, pool: List[Dict],
             results_file: str, format_pool_fn,
             dataset_key: str = "toxicity",
             model_name: str = "",
             eval_method: str = "toxigen") -> float:
    """
    Evaluate steered model (apply_yaml already patched before calling).
    Loads applier, generates, evaluates, cleans up.
    Returns score in [0, 100].
    """
    from steer.vector_appliers.vector_applier import BaseVectorApplier

    applier = BaseVectorApplier(cfg)
    applier.apply_vectors()
    if getattr(applier, "model", None) is not None:
        applier.model.set_from_positions(0)
    formatted = applier.generate(format_pool_fn(pool), save_results=False)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(formatted, f)
    if getattr(applier, "model", None) is not None:
        applier.model.reset_all()
    del applier
    torch.cuda.empty_cache()
    return run_eval(results_file, model_name, eval_method)
