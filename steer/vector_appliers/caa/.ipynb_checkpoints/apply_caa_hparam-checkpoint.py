"""
apply_caa_hparam.py
===================
MODIFIED FROM EASYEDIT2 ORIGINAL

Change 1  Added three fields for head-level steering:
            attention_heads: List[int]  — specific heads to target (None = all)
            use_head_steering: bool     — enable head-level steering
            injection_level: str        — "attention" (before FFN) or "block" (after FFN)
          Removed: vllm_enable, save_activations (these belong in the base
          HyperParams class, not here — adding them here shadows the base).

Bug fixed  attention_heads default was field(default_factory=lambda: None)
           which is invalid — default_factory must return a mutable default,
           not None. A scalar default should use default=None directly.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import yaml

from ...utils import HyperParams


@dataclass
class ApplyCAAHyperParams(HyperParams):
    # Model / steering config
    alg_name: str = "caa"
    layers: List[int] = field(default_factory=lambda: [24])
    multipliers: List[float] = field(default_factory=lambda: [1.0])
    steer_vector_load_dir: str = None

    # Change 1: head-level steering fields
    attention_heads: Optional[List[int]] = None          # None = block-level (default CAA)
    use_head_steering: bool = False                      # enable head-level steering
    injection_level: str = "attention"                   # "attention" or "block"

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config["alg_name"] == "caa") or print(
            f"ApplyCAAHyperParams cannot load from {hparams_name_or_path}, "
            f"alg_name is {config['alg_name']}"
        )
        return cls(**config)
