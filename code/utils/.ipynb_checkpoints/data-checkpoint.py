"""
utils/data.py
=============
Data splitting and pool formatting shared across all experiment scripts.
"""

import os
import json
import random
from typing import List, Dict, Optional

# ── Constants (overridable by scripts) ───────────────────────────────────────
SPLIT_SEED        = 42
TRAIN_VECTOR_FRAC = 0.80
EVAL_NUM_SAMPLES  = 200


def load_jsonl(path: str, cap: Optional[int] = None) -> List[Dict]:
    """Load a .jsonl file, optionally capping the number of items."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    if cap:
        data = data[:cap]
    return data


def make_splits(cfg, dataset_key: str = "toxicity",
                train_frac: float = TRAIN_VECTOR_FRAC,
                eval_cap: int = EVAL_NUM_SAMPLES,
                seed: int = SPLIT_SEED):
    """
    Split the training dataset into train and eval pools.

    Args:
        cfg:          OmegaConf config object (must have steer_train_dataset)
        dataset_key:  Key to look up in datasets dict e.g. "toxicity", "sst2_pair"
        train_frac:   Fraction of data used for vector training
        eval_cap:     Max samples in eval pool
        seed:         Random seed for shuffle

    Returns:
        train_data, eval_pool
    """
    from steer.datasets import prepare_train_dataset
    datasets = prepare_train_dataset(cfg)
    # Normalise dataset_key — OmegaConf may pass a ListConfig instead of str
    if not isinstance(dataset_key, str):
        dataset_key = list(dataset_key)[0] if hasattr(dataset_key, '__iter__') else str(dataset_key)
    # datasets dict key may differ — match by string value
    if dataset_key not in datasets:
        matched = [k for k in datasets
                   if str(k) == dataset_key or
                   (isinstance(k, (list, tuple)) and dataset_key in [str(x) for x in k])]
        if matched:
            data_list = datasets[matched[0]]
        else:
            raise KeyError(f"Dataset key '{dataset_key}' not found. "
                           f"Available: {list(datasets.keys())}")
    else:
        data_list = datasets[dataset_key]
    n         = len(data_list)
    rng       = random.Random(seed)
    idx       = list(range(n))
    rng.shuffle(idx)
    train_end  = int(train_frac * n)
    train_data = [data_list[i] for i in idx[:train_end]]
    eval_pool  = [data_list[i] for i in idx[train_end:]][:eval_cap]
    if len(eval_pool) < eval_cap:
        print(f"  Warning: eval pool only {len(eval_pool)} samples "
              f"(cap={eval_cap}, dataset may be too small)")
    print(f"  Total: {n}  Train: {len(train_data)}  Eval pool: {len(eval_pool)}")
    return train_data, eval_pool


def make_splits_with_holdout(dev_path: str,
                              cfg=None,
                              dataset_key: str = "toxicity",
                              train_frac: float = TRAIN_VECTOR_FRAC,
                              eval_cap: int = EVAL_NUM_SAMPLES,
                              holdout_cap: int = EVAL_NUM_SAMPLES,
                              seed: int = SPLIT_SEED):
    """
    Split training data into train/eval pools, and load a separate held-out
    dev set from dev_path (e.g. data/dev.jsonl).

    Returns:
        train_data, eval_pool, dev_pool
    """
    train_data, eval_pool = make_splits(
        cfg, dataset_key, train_frac, eval_cap, seed
    )
    dev_pool = load_jsonl(dev_path, cap=holdout_cap)
    print(f"  Dev pool:  {len(dev_pool)} samples  ({dev_path})")
    return train_data, eval_pool, dev_pool


def format_pool(pool: List[Dict],
                dataset_key: str = "toxicity",
                use_negative_prompt: bool = True) -> Dict:
    """
    Format a pool of dataset examples into the generation input format
    expected by BaseVectorApplier.generate().

    Args:
        pool:                 List of dataset examples
        dataset_key:          Output dict key e.g. "toxicity", "sst2_pair"
        use_negative_prompt:  If True, use "not_matching" as input (default for
                              toxicity — steer away from toxic completions).
                              If False, use "matching".

    Returns:
        {dataset_key: [{"input": ..., "reference_response": ...}, ...]}
    """
    out = []
    for ex in pool:
        if use_negative_prompt:
            text = (ex.get("not_matching")
                    or ex.get("input")
                    or ex.get("prompt"))
        else:
            text = (ex.get("matching")
                    or ex.get("input")
                    or ex.get("prompt"))
        if text:
            out.append({
                "input":              text,
                "reference_response": ex.get(
                    "matching" if use_negative_prompt else "not_matching", ""
                ),
            })
    return {dataset_key: out}