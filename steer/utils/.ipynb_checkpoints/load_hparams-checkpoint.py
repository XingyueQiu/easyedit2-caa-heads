"""
load_hparams.py
===============
MODIFIED FROM EASYEDIT2 ORIGINAL

Change 1  load_generate_vector_hparams: hparams_dict key changed from
          f"{alg_name}_{i}" to f"{alg_name}" — removes the index suffix
          so the key is just the algorithm name (e.g. "caa" not "caa_0").
          Note: this means only one set of hparams per alg_name is kept
          if the same algorithm appears more than once in the config.
          Acceptable for single-method experiments.

Change 2  load_generate_vector_hparams: added print(hparam_path) before
          loading to log which config file is being used.
"""

import os
from omegaconf import OmegaConf
from .alg_dict import HYPERPARAMS_CLASS_DICT, METHODS_CLASS_DICT


def load_generate_vector_hparams(top_cfg):
    hparams_dict = {}
    if isinstance(top_cfg.steer_vector_output_dirs, str):
        top_cfg.steer_vector_output_dirs = [top_cfg.steer_vector_output_dirs]

    for i, hparam_path in enumerate(top_cfg.steer_train_hparam_paths):
        assert os.path.exists(hparam_path), f"Hparam path {hparam_path} does not exist!"
        method_hparams = OmegaConf.load(hparam_path)
        print(hparam_path)  # Change 2
        alg_name = method_hparams["alg_name"]
        hparams_dict_key = f"{alg_name}"  # Change 1: was f"{alg_name}_{i}"
        combined_hparams = {**method_hparams, **top_cfg}
        selected_hparams_class = HYPERPARAMS_CLASS_DICT[alg_name]["train"]
        intersect_keys = (
            set(selected_hparams_class.__dataclass_fields__) & set(combined_hparams.keys())
        )
        hparams = selected_hparams_class(**{k: combined_hparams[k] for k in intersect_keys})
        print(f"Loading {alg_name} hparams from {hparam_path} ...")
        hparams.steer_vector_output_dir = (
            top_cfg.steer_vector_output_dirs[i]
            if i < len(top_cfg.steer_vector_output_dirs)
            else top_cfg.steer_vector_output_dirs[0]
        )
        hparams.steer_train_dataset = top_cfg.steer_train_dataset
        hparams_dict[hparams_dict_key] = hparams

    return hparams_dict


def load_apply_vector_hparams(top_cfg):
    hparams_dict = {}

    if not hasattr(top_cfg, "apply_steer_hparam_paths") or not top_cfg.apply_steer_hparam_paths:
        return hparams_dict

    for i, hparam_path in enumerate(top_cfg.apply_steer_hparam_paths):
        assert os.path.exists(hparam_path), f"Hparam path {hparam_path} does not exist!"
        method_hparams = OmegaConf.load(hparam_path)
        alg_name = method_hparams["alg_name"]
        combined_hparams = {**method_hparams, **top_cfg}
        selected_hparams_class = HYPERPARAMS_CLASS_DICT[alg_name]["apply"]
        intersect_keys = (
            set(selected_hparams_class.__dataclass_fields__) & set(combined_hparams.keys())
        )
        hparams = selected_hparams_class(**{k: combined_hparams[k] for k in intersect_keys})
        hparams.steer_vector_load_dir = top_cfg.steer_vector_load_dir[i]
        hparams_dict[alg_name] = hparams

    return hparams_dict
