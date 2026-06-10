"""
vector_generators.py (steer/vector_generators/vector_generators.py)
====================================================================
MODIFIED FROM EASYEDIT2 ORIGINAL

Change 1  REQUIRED_DATASET_METHODS: added 'reps', 'sft', 'spilt' upstream.
          Reverted to original list since those modules don't exist in your
          project. Also renamed 'reps_vector' → kept as-is from your version
          (your version uses 'reps_vector', original uses 'reps').

Change 2  generate_vectors: VLLM_SUPPORTED_METHODS was imported from
          alg_dict in the original (where it was defined as a module-level
          constant). In your version it was defined locally inside the
          method as an empty list [] — meaning vLLM was always disabled
          for all methods regardless of the constant in alg_dict.
          Since you removed VLLM_SUPPORTED_METHODS from alg_dict.py,
          keeping the local empty list is correct for your experiment
          (vLLM only supported for caa anyway).

Change 3  generate_vectors: dataset_name kwarg added to the train call
          for REQUIRED_DATASET_METHODS in the original. Your version did
          not pass dataset_name. Restored to pass it for forward
          compatibility (generate_caa_vectors accepts but ignores it).
"""

from omegaconf import DictConfig
import os

# Change 1: reps/sft/spilt removed — modules don't exist in your project
REQUIRED_DATASET_METHODS = ['lm_steer', 'caa', 'vector_prompt', 'sta', 'reps_vector']


class BaseVectorGenerator:
    """Base vector generator for all methods."""

    def __init__(self, top_cfg: DictConfig):
        from ..utils import load_generate_vector_hparams
        self.hparams_dict = load_generate_vector_hparams(top_cfg)
        for key, hparams in self.hparams_dict.items():
            print(f"{key.upper()} Generator Hyperparameters:\n{hparams}")

    def generate_vectors(self, datasets=None):
        from ..utils.seed import set_seed
        from ..utils.alg_dict import METHODS_CLASS_DICT

        assert datasets is not None, "Please provide datasets!"
        generated_vectors = {}

        for dataset_name in datasets:
            generated_vectors[dataset_name] = {}
            for key, hparams in self.hparams_dict.items():
                alg_name = hparams.alg_name
                if alg_name not in METHODS_CLASS_DICT:
                    raise NotImplementedError(f"Method {alg_name} not implemented!")

                set_seed(hparams.seed)
                steer_vector_output_dir = hparams.steer_vector_output_dir
                hparams.steer_vector_output_dir = os.path.join(
                    hparams.steer_vector_output_dir, dataset_name
                )
                now_path = os.path.join(
                    hparams.steer_vector_output_dir, alg_name + "_vector"
                )
                if os.path.exists(now_path) and hparams.save_vectors:
                    print("\033[1;34mVectors save path already exists! "
                          "The vector will be overwritten!\033[0m")

                print(f"Generating {key} vectors ...")
                if alg_name in REQUIRED_DATASET_METHODS:
                    # Change 2: local empty list — vLLM always disabled
                    VLLM_SUPPORTED_METHODS = []
                    if alg_name not in VLLM_SUPPORTED_METHODS:
                        hparams.vllm_enable = False
                    # Change 3: pass dataset_name kwarg
                    vectors = METHODS_CLASS_DICT[alg_name]["train"](
                        hparams, datasets[dataset_name], dataset_name=dataset_name
                    )
                else:
                    vectors = METHODS_CLASS_DICT[alg_name]["train"](hparams)

                generated_vectors[dataset_name][key] = vectors
                if hparams.save_vectors:
                    print(f"Saving vectors to {now_path} ...")
                else:
                    print(f"Not saving {key} vectors ...")
                hparams.steer_vector_output_dir = steer_vector_output_dir

        return generated_vectors

    def multimodal_generate_vectors(self, datasets=None):
        from ..utils.seed import set_seed
        from ..utils.alg_dict import METHODS_CLASS_DICT

        assert datasets is not None, "Please provide datasets!"
        generated_vectors = {}

        for dataset_name in datasets:
            generated_vectors[dataset_name] = {}
            for key, hparams in self.hparams_dict.items():
                alg_name = hparams.alg_name
                if alg_name not in METHODS_CLASS_DICT:
                    raise NotImplementedError(f"Method {alg_name} not implemented!")

                set_seed(hparams.seed)
                steer_vector_output_dir = hparams.steer_vector_output_dir
                hparams.steer_vector_output_dir = os.path.join(
                    hparams.steer_vector_output_dir, dataset_name
                )
                now_path = os.path.join(
                    hparams.steer_vector_output_dir, alg_name + "_vector"
                )
                if os.path.exists(now_path) and hparams.save_vectors:
                    print("\u001b[1;34mVectors save path already exists! "
                          "The vector will be overwritten!\u001b[0m")

                print(f"Generating {key} vectors ...")
                if alg_name in REQUIRED_DATASET_METHODS:
                    vectors = METHODS_CLASS_DICT[alg_name]["train"](
                        hparams, datasets[dataset_name], dataset_name=dataset_name
                    )
                else:
                    vectors = METHODS_CLASS_DICT[alg_name]["train"](hparams)

                generated_vectors[dataset_name][key] = vectors
                if hparams.save_vectors:
                    print(f"Saving vectors to {now_path} ...")
                else:
                    print(f"Not saving {key} vectors ...")
                hparams.steer_vector_output_dir = steer_vector_output_dir

        return generated_vectors
