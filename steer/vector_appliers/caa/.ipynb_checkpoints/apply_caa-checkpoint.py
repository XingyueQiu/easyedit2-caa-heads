"""
apply_caa.py
============
Apply CAA steering vectors to a loaded model wrapper.

MODIFIED FROM EASYEDIT2 ORIGINAL
Change 1  apply_caa(): added head-level steering branch — when
          hparams.use_head_steering=True and hparams.attention_heads is
          non-empty, routes to set_add_attn_activations_heads() with
          inject_at from hparams.injection_level (default "attention").
          Original called set_add_activations() unconditionally.

Bugs fixed vs both original and modified:
  - reset_caa_layers: operator precedence fixed on elif condition
    (A and B or C) → (A and (B or C)) to avoid AttributeError when
    model has no 'transformer' attribute
  - apply_caa: 'pipline' typo corrected to 'pipeline'
  - eval_caa: fully commented-out dead code removed
  - [APPLY_CAA] verbose debug prints removed from head-steering branch
"""

import os
import torch
from ...vector_generators.lm_steer import Hack_no_grad
from .apply_caa_hparam import ApplyCAAHyperParams


def reset_caa_layers(model, layers):
    """Reset CAA activations for the specified layers only."""
    model = model.model
    for layer in layers:
        # Llama / Qwen / Gemma family (model.model.layers)
        if hasattr(model, "model") and (
            hasattr(model.model, "layers")
            or (hasattr(model.model, "module") and hasattr(model.model.module, "layers"))
        ):
            if isinstance(model.model, Hack_no_grad):
                model.model.module.layers[layer].reset(method_name="caa")
            else:
                model.model.layers[layer].reset(method_name="caa")

        # GPT family (model.transformer.h)
        # Bug fix: was `A and B or C` — now `A and (B or C)` to avoid
        # AttributeError when model has no 'transformer' attribute
        elif hasattr(model, "transformer") and (
            hasattr(model.transformer, "h")
            or (hasattr(model.transformer, "module") and hasattr(model.transformer.module, "h"))
        ):
            if isinstance(model.transformer, Hack_no_grad):
                model.transformer.module.h[layer].reset(method_name="caa")
            else:
                model.transformer.h[layer].reset(method_name="caa")

        else:
            raise NotImplementedError(
                f"Cannot reset CAA activations for layer {layer}: "
                "unrecognised model structure (expected model.model.layers or model.transformer.h)"
            )


def apply_caa(hparams: ApplyCAAHyperParams, pipeline=None, vector=None):
    """
    Apply CAA steering vectors to `pipeline` (or load a model from hparams).

    Change 1: when hparams.use_head_steering=True and hparams.attention_heads
    is non-empty, applies head-level masked steering via
    set_add_attn_activations_heads(). Otherwise falls back to block-level
    set_add_activations() as in the original.
    """
    from ...models.get_model import get_model

    device = hparams.device
    if pipeline is None:
        model, _ = get_model(hparams)
    else:
        model = pipeline

    print(f"Applying CAA to model: {hparams.model_name_or_path}")
    reset_caa_layers(model, hparams.layers)

    for layer, multiplier in zip(hparams.layers, hparams.multipliers):
        if vector is not None:
            steering_vector = vector[f"layer_{layer}"].to(device)
        else:
            vector_path = os.path.join(
                hparams.steer_vector_load_dir, f"layer_{layer}.pt"
            )
            steering_vector = torch.load(vector_path, map_location=device)

        scaled_vector = multiplier * steering_vector

        # Change 1: head-level steering when enabled
        if hparams.use_head_steering and hparams.attention_heads:
            inject_at = getattr(hparams, "injection_level", "attention")
            model.set_add_attn_activations_heads(
                layer,
                scaled_vector,
                heads=hparams.attention_heads,
                method_name="caa",
                inject_at=inject_at,
            )
        else:
            model.set_add_activations(layer, scaled_vector, method_name="caa")

    return model
