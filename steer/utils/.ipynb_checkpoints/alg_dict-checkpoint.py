"""
alg_dict.py
===========
MODIFIED FROM EASYEDIT2 ORIGINAL

Change 1  Commented out / removed imports and registrations for methods
          not needed for your experiment:
            sae_feature, sta               — commented out in both dicts
            reps, sft, spilt               — removed entirely (these were
                                             added upstream after you cloned;
                                             the modules no longer exist in
                                             your project since they appear
                                             in the DELETED section of the
                                             audit report)

          Also removed VLLM_SUPPORTED_METHODS constant (upstream addition,
          not used in your scripts).

          What remains active: lm_steer, caa, vector_prompt, prompt,
          merge_vector — all you need for your experiment.
"""

from ..vector_generators import (
    VectorPromptHyperParams,
    CAAHyperParams,
    LmSteerHyperParams,
    MergeVectorHyperParams,
    # SaeFeatureHyperParams,  # not used
    # STAHyperParams,         # not used
)
from ..vector_appliers import (
    ApplyCAAHyperParams,
    ApplyVectorPromptHyperParams,
    ApplyLmSteerHyperParams,
    ApplyPromptHyperParams,
    ApplyMergeVectorHyperParams,
    # ApplySaeFeatureHyperParams,  # not used
    # ApplySTAHyperParams,         # not used
)
from ..vector_generators import (
    generate_lm_steer_delta,
    generate_caa_vectors,
    generate_vector_prompt_vectors,
    generate_merge_vector,
    # generate_sae_feature_vectors,  # not used
    # generate_sta_vectors,          # not used
)
from ..vector_appliers import (
    apply_lm_steer,
    apply_caa,
    apply_vector_prompt,
    apply_prompt,
    apply_merge_vector,
    # apply_sae_feature,  # not used
    # apply_sta,          # not used
)

import torch

DTYPES_DICT = {
    "float16":  torch.float16,
    "float32":  torch.float32,
    "float64":  torch.float64,
    "bfloat16": torch.bfloat16,
    "bf16":     torch.bfloat16,
    "fp16":     torch.float16,
    "fp32":     torch.float32,
    "fp64":     torch.float64,
}

HYPERPARAMS_CLASS_DICT = {
    "lm_steer":     {"train": LmSteerHyperParams,      "apply": ApplyLmSteerHyperParams},
    "caa":          {"train": CAAHyperParams,           "apply": ApplyCAAHyperParams},
    "vector_prompt":{"train": VectorPromptHyperParams,  "apply": ApplyVectorPromptHyperParams},
    "prompt":       {                                   "apply": ApplyPromptHyperParams},
    "merge_vector": {"train": MergeVectorHyperParams,   "apply": ApplyMergeVectorHyperParams},
    # "sae_feature":{"train": SaeFeatureHyperParams,    "apply": ApplySaeFeatureHyperParams},
    # "sta":        {"train": STAHyperParams,           "apply": ApplySTAHyperParams},
}

METHODS_CLASS_DICT = {
    "lm_steer":     {"train": generate_lm_steer_delta,         "apply": apply_lm_steer},
    "caa":          {"train": generate_caa_vectors,             "apply": apply_caa},
    "vector_prompt":{"train": generate_vector_prompt_vectors,   "apply": apply_vector_prompt},
    "prompt":       {                                           "apply": apply_prompt},
    "merge_vector": {"train": generate_merge_vector,            "apply": apply_merge_vector},
    # "sae_feature":{"train": generate_sae_feature_vectors,     "apply": apply_sae_feature},
    # "sta":        {"train": generate_sta_vectors,             "apply": apply_sta},
}
