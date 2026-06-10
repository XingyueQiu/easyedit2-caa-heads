"""
generate_sta_vectors.py
=======================
MODIFIED FROM EASYEDIT2 ORIGINAL

Change 1  generate_sta_vectors: added dataset_name parameter (unused,
          reserved for consistency with other generator signatures).
          All other logic is identical to the original.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
import os
from dotenv import load_dotenv

from .generate_sta_hparam import STAHyperParams
from typing import List

from ..sae_feature.sae_utils import (
    clear_gpu_cache,
    load_sae_from_dir,
    load_gemma_2_sae,
)

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


def prepare_input(tokenizer, prompts, device="cuda"):
    input_tokens = tokenizer.tokenize(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(device)
    return input_tokens


def signed_min_max_normalize(tensor):
    abs_tensor = tensor.abs()
    min_val = abs_tensor.min()
    max_val = abs_tensor.max()
    normalized = (abs_tensor - min_val) / (max_val - min_val)
    return tensor.sign() * normalized


def act_and_fre(act_data, pos_data, neg_data, mode, trim, sae):
    pec = 1 - trim

    act_data_init = act_data.to(sae.W_dec.device)
    diff_data = pos_data - neg_data

    norm_act  = signed_min_max_normalize(act_data)
    norm_diff = signed_min_max_normalize(diff_data)

    mask = ((norm_act > 0) & (norm_diff > 0)) | ((norm_act < 0) & (norm_diff < 0))
    print("mask:", mask.sum())

    scores = torch.zeros_like(norm_diff)
    scores[mask] = norm_diff[mask]

    threshold_fre = torch.sort(torch.abs(scores), descending=True, stable=True).values[
        int(pec * len(scores))
    ]
    print(f"frequency threshold: {threshold_fre}")
    freq_mask = torch.abs(scores) >= threshold_fre
    print("freq_mask:", freq_mask.sum())

    threshold = torch.sort(torch.abs(act_data_init), descending=True, stable=True).values[
        int(pec * len(act_data_init))
    ]
    print(f"threshold: {threshold}")
    act_top_mask = torch.abs(act_data_init) >= threshold
    print("act_top_mask:", act_top_mask.sum())

    result = None
    if mode == "act_and_freq":
        freq_mask = freq_mask.to(sae.W_dec.device)
        act_data_combined = act_data.clone()
        combined_mask = freq_mask & act_top_mask
        print("combined_mask:", combined_mask.sum())
        act_data_combined[~combined_mask] = 0
        print(torch.abs(act_data_combined).sum())
        act_data_combined = act_data_combined.to(sae.W_dec.device)
        result = act_data_combined @ sae.W_dec
        print("result_combined.shape", result.shape)
        print("result_combined:", result)
        print("torch.norm(result_combined)", torch.norm(result))
    elif mode == "only_act":
        act_data_act = act_data.clone()
        act_data_act[~act_top_mask] = 0
        act_data_act = act_data_act.to(sae.W_dec.device)
        result = act_data_act @ sae.W_dec
    elif mode == "only_freq":
        act_data_fre = act_data.clone()
        act_data_fre[~freq_mask] = 0
        act_data_fre = act_data_fre.to(sae.W_dec.device)
        result = act_data_fre @ sae.W_dec

    return result


def generate_sta_vectors(hparams: STAHyperParams, dataset,
                          model=None, dataset_name=None):  # Change 1
    from ...models.get_model import get_model
    from ...datasets.caa_data import get_tokens_for_caa

    args = hparams
    del_model = True
    assert len(args.layers) == 1, "Not support many layers!!!"
    assert len(args.layers) == len(args.sae_paths), \
        f"len(sae_paths) does not match the len(layers)"
    for i, layer in enumerate(args.layers):
        assert str(args.layers[i]) in args.sae_paths[i], \
            f"Saes[{i}] does not match the layers[{i}]!!!"

    if model is None:
        model, tokenizer = get_model(hparams)
    else:
        del_model = False
        model, tokenizer = model, model.tokenizer
        model.hparams = hparams

    device = model.device
    saes = {layer: [] for layer in args.layers}
    for i, layer in enumerate(args.layers):
        assert os.path.exists(args.sae_paths[i]), \
            f"{args.sae_paths[i]} does not exist!!!"
        if "gemma" in args.model_name_or_path.lower():
            saes[layer], _, _ = load_gemma_2_sae(sae_path=args.sae_paths[i], device=device)
        else:
            saes[layer], _, _ = load_sae_from_dir(args.sae_paths[i], device=device)

    need_train_layers = []
    vectors = {}
    for layer in args.layers:
        feature_act_path = os.path.join(
            args.steer_vector_output_dir, f"feature_score/layer_{layer}_feature_act.pt")
        pos_freq_path = os.path.join(
            args.steer_vector_output_dir, f"feature_score/layer_{layer}_pos_freq.pt")
        neg_freq_path = os.path.join(
            args.steer_vector_output_dir, f"feature_score/layer_{layer}_neg_freq.pt")
        caa_vector_path = os.path.join(
            args.steer_vector_output_dir,
            f"caa_vector/layer_{layer}.pt"
            if not args.multiple_choice
            else f"caa_vector_multiple_choice/layer_{layer}.pt"
        )

        if all(os.path.exists(p) for p in
               [feature_act_path, pos_freq_path, neg_freq_path, caa_vector_path]):
            feature_act      = torch.load(feature_act_path, map_location=device)
            pos_feature_freq = torch.load(pos_freq_path,    map_location=device)
            neg_feature_freq = torch.load(neg_freq_path,    map_location=device)
            caa_vector       = torch.load(caa_vector_path,  map_location=device)

            if args.save_vectors:
                output_dir_sta = os.path.join(
                    args.steer_vector_output_dir,
                    "sta_vector" if not args.multiple_choice else "sta_vector_multiple_choice"
                )
                os.makedirs(output_dir_sta, exist_ok=True)

            for trim in args.trims:
                sta_vec = act_and_fre(
                    feature_act, pos_feature_freq, neg_feature_freq,
                    args.mode, trim, saes[layer]
                )
                caa_vector = caa_vector.to(device)
                sta_vec    = sta_vec.to(device)
                multiplier = torch.norm(caa_vector, p=2) / torch.norm(sta_vec, p=2)
                print(f"caa_norm:{caa_vector.norm()}  sta_norm:{(multiplier * sta_vec).norm()}")
                if args.save_vectors:
                    torch.save(
                        multiplier * sta_vec,
                        os.path.join(output_dir_sta, f"layer_{layer}_{args.mode}_trim{trim}.pt"),
                    )
                vectors[f"layer_{layer}_{args.mode}_trim{trim}"] = multiplier * sta_vec
        else:
            need_train_layers.append(layer)

    if not need_train_layers:
        print(f"Computed vector using historical saved data: "
              f"{os.path.join(args.steer_vector_output_dir, 'feature_score')}")
        return vectors

    args.layers = need_train_layers
    pos_activations     = {layer: [] for layer in args.layers}
    neg_activations     = {layer: [] for layer in args.layers}
    pos_sae_activations = {layer: [] for layer in args.layers}
    neg_sae_activations = {layer: [] for layer in args.layers}

    pos_tokens_list, neg_tokens_list = get_tokens_for_caa(dataset, tokenizer, hparams)

    for p_tokens_dict, n_tokens_dict in tqdm(
        zip(pos_tokens_list, neg_tokens_list),
        total=len(pos_tokens_list),
        desc="Processing prompts",
    ):
        p_tokens        = p_tokens_dict["pos_tokens"]
        n_tokens        = n_tokens_dict["neg_tokens"]
        ques_tokens_len = p_tokens_dict["ques_tokens_len"]

        model.reset_all()
        model.get_logits(p_tokens)
        for layer in args.layers:
            p_act     = model.get_last_activations(layer)
            p_sae_act = saes[layer].encode(p_act)
            if args.multiple_choice:
                p_act     = p_act[0, -2, :].detach().cpu()
                p_sae_act = p_sae_act[0, -2, :].detach().cpu()
            else:
                p_act     = p_act[0, ques_tokens_len:, :].mean(0).detach().cpu()
                p_sae_act = p_sae_act[0, ques_tokens_len:, :].mean(0).detach().cpu()
            pos_activations[layer].append(p_act)
            pos_sae_activations[layer].append(p_sae_act)

        model.reset_all()
        model.get_logits(n_tokens)
        for layer in args.layers:
            n_act     = model.get_last_activations(layer)
            n_sae_act = saes[layer].encode(n_act)
            if args.multiple_choice:
                n_act     = n_act[0, -2, :].detach().cpu()
                n_sae_act = n_sae_act[0, -2, :].detach().cpu()
            else:
                n_act     = n_act[0, ques_tokens_len:, :].mean(0).detach().cpu()
                n_sae_act = n_sae_act[0, ques_tokens_len:, :].mean(0).detach().cpu()
            neg_activations[layer].append(n_act)
            neg_sae_activations[layer].append(n_sae_act)

    if args.save_vectors:
        os.makedirs(args.steer_vector_output_dir, exist_ok=True)

    for layer in args.layers:
        all_pos = torch.stack(pos_activations[layer])
        all_neg = torch.stack(neg_activations[layer])
        caa_vector = (all_pos - all_neg).mean(dim=0)

        if args.save_vectors:
            output_dir_caa = os.path.join(
                args.steer_vector_output_dir,
                "caa_vector" if not args.multiple_choice else "caa_vector_multiple_choice"
            )
            os.makedirs(output_dir_caa, exist_ok=True)
            torch.save(caa_vector, os.path.join(output_dir_caa, f"layer_{layer}.pt"))
        vectors[f"layer_{layer}"] = caa_vector

        all_pos_sae = torch.stack(pos_sae_activations[layer])
        all_neg_sae = torch.stack(neg_sae_activations[layer])
        feature_act      = (all_pos_sae - all_neg_sae).mean(dim=0)
        pos_feature_freq = (all_pos_sae > 0).float().sum(0)
        neg_feature_freq = (all_neg_sae > 0).float().sum(0)

        if args.save_vectors:
            score_dir = os.path.join(args.steer_vector_output_dir, "feature_score")
            os.makedirs(score_dir, exist_ok=True)
            torch.save(pos_feature_freq, os.path.join(score_dir, f"layer_{layer}_pos_freq.pt"))
            torch.save(neg_feature_freq, os.path.join(score_dir, f"layer_{layer}_neg_freq.pt"))
            torch.save(feature_act,      os.path.join(score_dir, f"layer_{layer}_feature_act.pt"))

        if args.save_vectors:
            output_dir_sta = os.path.join(
                args.steer_vector_output_dir,
                "sta_vector" if not args.multiple_choice else "sta_vector_multiple_choice"
            )
            os.makedirs(output_dir_sta, exist_ok=True)

        try:
            for trim in args.trims:
                sta_vec = act_and_fre(
                    feature_act, pos_feature_freq, neg_feature_freq,
                    args.mode, trim, saes[layer]
                )
                caa_vector = caa_vector.to(device)
                sta_vec    = sta_vec.to(device)
                multiplier = torch.norm(caa_vector, p=2) / torch.norm(sta_vec, p=2)
                print(f"caa_norm:{caa_vector.norm()}  sta_norm:{(multiplier * sta_vec).norm()}")
                if args.save_vectors:
                    torch.save(
                        multiplier * sta_vec,
                        os.path.join(output_dir_sta, f"layer_{layer}_{args.mode}_trim{trim}.pt"),
                    )
                vectors[f"layer_{layer}_{args.mode}_trim{trim}"] = multiplier * sta_vec
        finally:
            if del_model:
                model.model.to("cpu")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams_path", type=str, default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=[24])
    parser.add_argument("--sae_paths", nargs="+", type=str,
                        default=["/data2/xzwnlp/gemma-scope-9b-pt-res/layer_24/width_16k/average_l0_114"])
    parser.add_argument("--trims", nargs="+", type=float, default=[0.65])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, default="act_and_freq")
    parser.add_argument("--dataset_name", type=str, default="safeedit")
    parser.add_argument("--model_name_or_path", type=str,
                        default="/mnt/20t/msy/models/gemma-2-9b-base")
    parser.add_argument("--multiple_choice", action="store_true", default=False)
    parser.add_argument("--steer_vector_output_dir", type=str, default="../")
    args = parser.parse_args()

    if args.hparams_path:
        hparams = STAHyperParams.from_hparams(args.hparams_path)
    else:
        hparams = STAHyperParams(
            model_name_or_path=args.model_name_or_path,
            layers=args.layers,
            sae_paths=args.sae_paths,
            device=args.device,
            dataset_name=args.steer_train_dataset,
            trims=args.trims,
            mode=args.mode,
            multiple_choice=args.multiple_choice,
            steer_vector_output_dir=args.steer_vector_output_dir,
        )
    generate_sta_vectors(hparams)
