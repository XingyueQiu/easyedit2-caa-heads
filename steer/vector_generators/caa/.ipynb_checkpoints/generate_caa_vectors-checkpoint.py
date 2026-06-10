"""
generate_caa_vectors.py
=======================
MODIFIED FROM EASYEDIT2 ORIGINAL

Change 1  generate_caa_vectors: added dataset_name parameter (unused
          currently but reserved for dataset-specific logic).

Change 2  Activation extraction: added None-check for get_last_activations()
          result and added handling for 2-D activations (dim==2 path).
          Original assumed activations were always 3-D and never None.

Change 3  Vector save loop: added print statements confirming save path
          and success, and wrapped in try/except with traceback on failure.
          Original used try/finally with no except — a failed save would
          silently swallow the error and return an empty dict.

Bug fixed  return vectors was inside the finally block in the original.
           In Python, a return inside finally executes unconditionally,
           meaning even if the try block raised an exception, vectors
           (possibly empty or partial) would still be returned silently
           instead of propagating the error. Moved return outside finally.
"""

import sys
sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

import os
import torch
import argparse
import traceback
from tqdm import tqdm
from dotenv import load_dotenv

from .generate_caa_hparam import CAAHyperParams

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


def generate_caa_vectors(hparams: CAAHyperParams, dataset, model=None,
                          dataset_name=None):  # Change 1: added dataset_name
    from ...models.get_model import get_model
    from ...datasets.caa_data import get_tokens_for_caa

    del_model = True
    args = hparams

    if model is None:
        model, tokenizer = get_model(hparams)
    else:
        del_model = False
        model, tokenizer = model, model.tokenizer
        model.hparams = hparams

    pos_activations = {layer: [] for layer in args.layers}
    neg_activations = {layer: [] for layer in args.layers}

    pos_tokens_list, neg_tokens_list = get_tokens_for_caa(dataset, tokenizer, hparams)

    for p_tokens_dict, n_tokens_dict in tqdm(
        zip(pos_tokens_list, neg_tokens_list),
        total=len(pos_tokens_list),
        desc="Processing prompts",
    ):
        p_tokens       = p_tokens_dict["pos_tokens"]
        n_tokens       = n_tokens_dict["neg_tokens"]
        ques_tokens_len = p_tokens_dict["ques_tokens_len"]

        model.reset_all()
        model.get_logits(p_tokens)

        for layer in args.layers:
            p_activations = model.get_last_activations(layer)
            if p_activations is None:  # Change 2: None guard
                continue
            if args.multiple_choice:
                p_activations = p_activations[0, -2, :].detach().cpu()
            elif p_activations.dim() == 3:  # Change 2: explicit dim checks
                p_activations = p_activations[0, ques_tokens_len:, :].mean(0).detach().cpu()
            elif p_activations.dim() == 2:
                p_activations = p_activations[ques_tokens_len:, :].mean(0).detach().cpu()
            else:
                raise ValueError(f"Unexpected activation shape: {p_activations.shape}")
            pos_activations[layer].append(p_activations)

        model.reset_all()
        model.get_logits(n_tokens)

        ques_tokens_len = n_tokens_dict["ques_tokens_len"]
        for layer in args.layers:
            n_activations = model.get_last_activations(layer)
            if n_activations is None:  # Change 2: None guard
                continue
            if args.multiple_choice:
                n_activations = n_activations[0, -2, :].detach().cpu()
            elif n_activations.dim() == 3:
                n_activations = n_activations[0, ques_tokens_len:, :].mean(0).detach().cpu()
            elif n_activations.dim() == 2:
                n_activations = n_activations[ques_tokens_len:, :].mean(0).detach().cpu()
            else:
                raise ValueError(f"Unexpected activation shape: {n_activations.shape}")
            neg_activations[layer].append(n_activations)

    if hparams.save_vectors:
        subdir = "caa_vector_multiple_choice" if args.multiple_choice else "caa_vector"
        steer_vector_output_dir = os.path.join(args.steer_vector_output_dir, subdir)
        os.makedirs(steer_vector_output_dir, exist_ok=True)

    # Change 3: try/except with traceback; return outside finally (bug fix)
    try:
        vectors = {}
        for layer in args.layers:
            all_pos_layer = torch.stack(pos_activations[layer])
            all_neg_layer = torch.stack(neg_activations[layer])
            vec = (all_pos_layer - all_neg_layer).mean(dim=0)
            if hparams.save_vectors:
                save_path = os.path.join(steer_vector_output_dir, f"layer_{layer}.pt")
                print(f"Saving to: {save_path}")
                torch.save(vec, save_path)
                print("Saved successfully.")
            vectors[f"layer_{layer}"] = vec
    except Exception as e:
        print(f"ERROR during vector computation/save: {e}")
        traceback.print_exc()
        vectors = {}
    finally:
        if del_model:
            model.model.to("cpu")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return vectors  # Bug fix: was inside finally block in original


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams_path", type=str, default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_path", type=str,
                        default="/mnt/16t/xzwnlp/SaeEdit/git/shae/data/safety/safeedit")
    parser.add_argument("--model_name_or_path", type=str,
                        default="/mnt/20t/msy/models/gemma-2-9b-base")
    parser.add_argument("--multiple_choice", action="store_true", default=False)
    parser.add_argument("--steer_vector_output_dir", type=str, default="../")
    args = parser.parse_args()

    if args.hparams_path:
        hparams = CAAHyperParams.from_hparams(args.hparams_path)
    else:
        hparams = CAAHyperParams(
            layers=args.layers,
            data_path=args.data_path,
            model_name_or_path=args.model_name_or_path,
            device=args.device,
            multiple_choice=args.multiple_choice,
            steer_vector_output_dir=args.steer_vector_output_dir,
        )

    generate_caa_vectors(hparams)
