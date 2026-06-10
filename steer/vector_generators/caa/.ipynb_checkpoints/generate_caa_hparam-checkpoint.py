"""
generate_caa_hparam.py
======================
MODIFIED FROM EASYEDIT2 ORIGINAL

Change 1  Added save_activations: bool = True field.
          Now redundant — save_activations has been moved to the base
          HyperParams class. This field can be removed here entirely
          without breaking anything, since the base class already provides
          it. Kept for now to avoid any downstream surprises, but should
          be cleaned up once all hparam subclasses are confirmed to
          inherit from the updated base.
"""

import yaml
from typing import List
from dataclasses import dataclass, field

from ...utils import HyperParams


@dataclass
class CAAHyperParams(HyperParams):
    alg_name: str = "caa"
    layers: List[int] = field(default_factory=lambda: list(range(32)))
    steer_train_dataset: str = "safeedit"
    multiple_choice: bool = False
    steer_vector_output_dir: str = "../"
    save_vectors: bool = True
    # Change 1: added here — now redundant since base HyperParams has it
    # Can be removed once base class update is confirmed propagated
    save_activations: bool = True

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config["alg_name"] == "caa") or print(
            f"CAAHyperParams cannot load from {hparams_name_or_path}, "
            f"alg_name is {config['alg_name']}"
        )
        return cls(**config)
