"""
hparams.py
==========
MODIFIED FROM EASYEDIT2 ORIGINAL

Change 1  Added two fields to base class:
            vllm_enable: bool = False
            save_activations: bool = True
          These were originally added directly to ApplyCAAHyperParams
          (wrong place — they shadow base class values). Moving them here
          makes them available to all hparam subclasses consistently.
"""

import json
from dataclasses import dataclass, asdict


@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """
    use_chat_template: bool = False
    system_prompt: str = ""
    torch_dtype: str = "fp32"
    seed: int = 42
    model_name_or_path: str = "your_own_path"
    device: str = "cpu"
    use_cache: bool = True
    generate_orig_output: bool = False

    # Change 1: vLLM and activation saving flags
    vllm_enable: bool = False
    save_activations: bool = True

    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)
        return cls(**data)

    def construct_float_from_scientific_notation(config: dict):
        for key, value in config.items():
            if isinstance(value, str):
                try:
                    config[key] = float(value)
                except Exception:
                    pass
        return config

    def to_dict(config) -> dict:
        return asdict(config)
