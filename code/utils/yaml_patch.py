"""
utils/yaml_patch.py
===================
Config loading, model introspection, yaml patching, and vector file
utilities shared across all experiment scripts.
"""

import os
import yaml
import torch
from typing import Optional
from omegaconf import OmegaConf
from transformers import AutoConfig


# ── Config helpers ────────────────────────────────────────────────────────────

def first_path(v) -> Optional[str]:
    """Extract a single string path from a string or list."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    return str(v[0]) if len(v) > 0 else None


def load_config(config_path: str):
    """
    Load OmegaConf config and normalise steer_vector_output_dirs /
    steer_vector_load_dir to single-element lists as expected by
    BaseVectorApplier and BaseVectorGenerator.
    """
    cfg      = OmegaConf.load(config_path)
    out_dir  = first_path(cfg.get("steer_vector_output_dirs"))
    load_dir = first_path(cfg.get("steer_vector_load_dir"))
    if not load_dir: load_dir = out_dir
    if not out_dir:  out_dir  = load_dir
    cfg.steer_vector_output_dirs = [out_dir]
    cfg.steer_vector_load_dir    = [load_dir]
    return cfg


def load_config_raw(config_path: str):
    """
    Load OmegaConf config without modification.
    Used for BaseVectorGenerator which expects unmodified paths.
    """
    return OmegaConf.load(config_path)


# ── Model introspection ───────────────────────────────────────────────────────

def get_n_layers(model_name: str) -> int:
    """Return number of transformer layers for the given model."""
    c = AutoConfig.from_pretrained(model_name)
    return getattr(c, "num_hidden_layers", getattr(c, "n_layer", 12))


def get_n_heads(model_name: str) -> int:
    """Return number of attention heads for the given model."""
    c = AutoConfig.from_pretrained(model_name)
    return getattr(c, "num_attention_heads", getattr(c, "n_head", 12))


# ── YAML patching ─────────────────────────────────────────────────────────────

def set_generate_yaml_layer(generate_yaml: str, layer: int):
    """Patch generate_caa.yaml to target a specific layer."""
    with open(generate_yaml) as f:
        g = yaml.safe_load(f)
    g["layers"] = [layer]
    with open(generate_yaml, "w") as f:
        yaml.dump(g, f)


def set_apply_yaml(apply_yaml: str, generate_yaml: str,
                   layer: int, mu: float,
                   use_head_steering: bool = False,
                   attention_heads: Optional[list] = None):
    """
    Patch apply_caa.yaml and generate_caa.yaml for a given layer and mu.

    Args:
        apply_yaml:        Path to apply_caa.yaml
        generate_yaml:     Path to generate_caa.yaml
        layer:             Target layer index
        mu:                Steering multiplier
        use_head_steering: Enable head-level steering
        attention_heads:   List of head indices (None or [] = all heads / block)
    """
    with open(apply_yaml)    as f: a = yaml.safe_load(f)
    with open(generate_yaml) as f: g = yaml.safe_load(f)

    heads = attention_heads if attention_heads is not None else []

    for d in (a, g):
        d["layers"]            = [layer]
        d["use_head_steering"] = use_head_steering
        d["attention_heads"]   = heads

    a["multipliers"] = [float(mu)]

    with open(apply_yaml,    "w") as f: yaml.dump(a, f)
    with open(generate_yaml, "w") as f: yaml.dump(g, f)


# ── Vector file helpers ───────────────────────────────────────────────────────

def find_vector_file(vec_dir: str, layer: int) -> Optional[str]:
    """
    Find the saved vector file for a layer under vec_dir.
    Checks common EasyEdit save path patterns, then falls back to
    a recursive search.
    """
    candidates = [
        os.path.join(vec_dir, f"layer_{layer}.pt"),
        os.path.join(vec_dir, "toxicity",  "caa_vector", f"layer_{layer}.pt"),
        os.path.join(vec_dir, "sst2_pair", "caa_vector", f"layer_{layer}.pt"),
        os.path.join(vec_dir, "caa_vector",               f"layer_{layer}.pt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    # Recursive fallback
    for root, _, files in os.walk(vec_dir):
        if f"layer_{layer}.pt" in files:
            return os.path.join(root, f"layer_{layer}.pt")
    return None


def ensure_vector(vec_dir: str, layer: int, train_data,
                  config_path: str, generate_yaml: str,
                  dataset_key: str = "toxicity") -> str:
    """
    Return path to vector file for layer, training it first if missing.

    Args:
        vec_dir:      Directory to search for vectors
        layer:        Layer index
        train_data:   Training examples (used if vector must be trained)
        config_path:  Path to experiment config yaml
        generate_yaml: Path to generate_caa.yaml (will be patched)
        dataset_key:  Dataset key for BaseVectorGenerator

    Returns:
        Absolute path to the vector file.
    """
    from steer.vector_generators.vector_generators import BaseVectorGenerator

    path = find_vector_file(vec_dir, layer)
    if path:
        vec = torch.load(path, map_location="cpu")
        print(f"  layer_{layer}.pt found: {path}  "
              f"shape={vec.shape}  norm={vec.norm().item():.4f}")
        return path

    print(f"  layer_{layer}.pt not found — training now ...")
    set_generate_yaml_layer(generate_yaml, layer)
    cfg_raw   = load_config_raw(config_path)
    generator = BaseVectorGenerator(cfg_raw)
    generator.generate_vectors({dataset_key: train_data})

    path = find_vector_file(vec_dir, layer)
    if not path:
        raise FileNotFoundError(
            f"Vector not found after training for layer {layer}. "
            f"Check EasyEdit save path logic under {vec_dir}."
        )
    vec = torch.load(path, map_location="cpu")
    print(f"  Trained and saved: {path}  "
          f"shape={vec.shape}  norm={vec.norm().item():.4f}")
    return path
