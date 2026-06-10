"""
evaluate.py
===========
Evaluation utilities for CAA steering experiments.

MODIFIED FROM EASYEDIT2 ORIGINAL
Changes vs original:
  Change 1  BASE_URL: added default fallback value 'https://api.gpt.ge/v1/'
  Change 2  Added openai.default_headers = {"x-foo": "true"}
  Change 3  _evaluate_toxigen: switched model from local path to HuggingFace
              'tomh/toxigen_roberta'
  Change 4  _evaluate_toxigen: switched from binary _score_generations to
              continuous _score_probabilities for finer resolution
  Change 5  _score_probabilities: new method added (continuous P(toxic) scoring)

Bugs fixed in this cleaned version (were present in both original and modified):
  - _evaluate_realtoxicityprompts: generations_df was a plain list but
    .iterrows()/.index (pandas API) were called on it → fixed to enumerate()
  - _score_probabilities: missing @torch.no_grad() decorator → added
  - _evaluate_toxigen: dead commented-out local path and old call removed
  - _evaluate_safeedit: hardcoded absolute path made into constructor param
  - total_failed: declared but never updated → removed
  - HttpError: imported but never used → removed
  - evaluate_all / evaluate_from_file: duplicated logic unified

Removed (not needed for toxicity/SST2 experiment):
  - _evaluate_gsm          (GSM8K math benchmark, unrelated)
  - _evaluate_safeedit     (SafeEdit classifier, unrelated + broken path)
  - _evaluate_realtoxicityprompts  (Perspective API, external dependency)
  - _llm_evaluate          (GPT-4 judge, external API dependency)
  - _harmonic_mean         (only used by _llm_evaluate)
  - _score_generations     (superseded by _score_probabilities)
  All openai / googleapiclient / asyncio imports removed with them.
"""

import argparse
import json
import os
import sys

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")

import torch
import numpy as np
import nltk
import scipy
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    AutoModelForSequenceClassification,
)
from typing import List, Dict
import warnings


# HuggingFace model ID for toxicity classifier
# Change 3: was '/mnt/16t/xzwnlp/model/toxigen_roberta' (local absolute path)
TOXIGEN_MODEL_ID = os.environ.get("TOXIGEN_MODEL_PATH", "tomh/toxigen_roberta")


class Evaluator:
    def __init__(self, **kwargs):
        self.mode              = kwargs.get("mode", "file")
        self.save_results      = kwargs.get("save_results", True)
        self.device            = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model_name_or_path = kwargs.get("model_name_or_path", None)
        self.eval_methods      = kwargs.get("eval_methods", None)
        self.dataset_path      = kwargs.get("generation_dataset_path", None)

        if self.save_results:
            self.results_dir = kwargs.get("results_dir", "results")

    # ── Public interface ─────────────────────────────────────────────────────

    def evaluate_from_file(self, dataset_path: str, concept: str = None):
        file_path = dataset_path or self.dataset_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Results file not found: {file_path}")

        dataset_name = os.path.basename(file_path).replace("_results.json", "")
        print(f"\nEvaluating results for dataset: {dataset_name}")

        with open(file_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        return self._run_evaluation(results, dataset_name, concept)

    def evaluate_from_direct(self, results: List[Dict], dataset_name: str,
                              concept: str = None):
        print(f"\nEvaluating results directly for: {dataset_name}")
        return self._run_evaluation(results, dataset_name, concept)

    def evaluate_all(self, concept: str = None):
        """Evaluate from self.dataset_path. Serves as the CLI entry point."""
        return self.evaluate_from_file(self.dataset_path, concept)

    # ── Core dispatcher ──────────────────────────────────────────────────────

    def evaluate(self, results: List[Dict], dataset_name: str,
                 concept: str = None) -> Dict:
        eval_results = {}

        for method in self.eval_methods:
            print(f"Running evaluation method: {method}")
            m = method.lower()

            if "ppl" in m:
                with torch.no_grad():
                    ppl, total_ppl = self._calc_perplexity(results)
                eval_results["perplexity"]       = ppl
                eval_results["total_perplexity"] = total_ppl

            elif "sentiment" in m:
                neg_acc, neg_std, pos_acc, pos_std = self._calc_sentiment(results)
                eval_results["mean_negative_sentiment"] = neg_acc
                eval_results["std_negative_sentiment"]  = neg_std
                eval_results["mean_positive_sentiment"] = pos_acc
                eval_results["std_positive_sentiment"]  = pos_std

            elif "distinctness" in m:
                dist1, dist2, dist3 = self._calc_distinctness(results)
                eval_results["dist-1"] = dist1
                eval_results["dist-2"] = dist2
                eval_results["dist-3"] = dist3

            elif "toxigen" in m:
                toxigen_results = self._evaluate_toxigen(results)
                eval_results.update(toxigen_results)

            elif "fluency" in m:
                texts = [text for item in results for text in item["pred"]]
                eval_results["fluency"] = float(np.mean(self._n_gram_entropy(texts)))

            else:
                warnings.warn(f"Unknown eval method '{method}' — skipped")

            print(f"Current results: {eval_results}\n")

        return eval_results

    # ── Internal helper ──────────────────────────────────────────────────────

    def _run_evaluation(self, results: List[Dict], dataset_name: str,
                        concept: str = None) -> Dict:
        eval_results = self.evaluate(results, dataset_name, concept)
        if self.save_results:
            out_path = os.path.join(
                self.results_dir, f"{dataset_name}_evaluation.json"
            )
            self.save_all_results(eval_results, out_path)
            print(f"Evaluation results saved to: {out_path}")
        return eval_results

    def save_all_results(self, results: Dict, output_file: str):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    # ── Evaluation methods ───────────────────────────────────────────────────

    def _calc_perplexity(self, results: List[Dict]):
        ppl_model     = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path).to(self.device)
        ppl_tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        perplexities = []
        total_nll    = 0
        total_tokens = 0

        for item in tqdm(results, desc="Evaluating PPL"):
            prompt = item["input"]
            prompt_ids = ppl_tokenizer.encode(
                prompt, return_tensors="pt").to(self.device)

            if not (prompt_ids.shape[1] == 1
                    and prompt_ids[0][0] == ppl_tokenizer.bos_token_id):
                prompt_loss = (ppl_model(prompt_ids, labels=prompt_ids)[0]
                               * (prompt_ids.shape[1] - 1))
            else:
                prompt_loss = 0

            for gen in item["pred"]:
                full_ids = ppl_tokenizer.encode(
                    f"{prompt}{gen}", return_tensors="pt").to(self.device)
                full_loss = (ppl_model(full_ids, labels=full_ids)[0]
                             * (full_ids.shape[1] - 1))
                gen_tokens = full_ids.shape[1] - prompt_ids.shape[1]
                if gen_tokens <= 0:
                    continue
                loss = (full_loss - prompt_loss) / gen_tokens
                ppl  = np.exp(loss.item())
                if ppl < 1e4:
                    perplexities.append(ppl)
                total_nll    += (full_loss - prompt_loss).item()
                total_tokens += gen_tokens

        mean_ppl   = float(np.nanmean(perplexities))
        corpus_ppl = float(np.exp(total_nll / total_tokens)) if total_tokens > 0 else float("inf")
        return mean_ppl, corpus_ppl

    def _calc_sentiment(self, results: List[Dict]):
        classifier    = pipeline("sentiment-analysis", device=self.device)
        neg_accuracies = []
        pos_accuracies = []

        for item in tqdm(results, desc="Scoring sentiments"):
            prompt = item["input"]
            for gen in item["pred"]:
                try:
                    sentence   = f"{prompt}{gen}"
                    prediction = classifier([sentence], max_length=512)[0]
                    label      = prediction["label"].upper()
                    neg_accuracies.append(float(label == "NEGATIVE"))
                    pos_accuracies.append(float(label == "POSITIVE"))
                except IndexError:
                    neg_accuracies.append(float("nan"))
                    pos_accuracies.append(float("nan"))

        return (float(np.nanmean(neg_accuracies)), float(np.std(neg_accuracies)),
                float(np.nanmean(pos_accuracies)), float(np.std(pos_accuracies)))

    def _calc_distinctness(self, results: List[Dict]):
        dist1, dist2, dist3 = [], [], []

        for item in tqdm(results, desc="Evaluating dist-n"):
            unigrams, bigrams, trigrams = set(), set(), set()
            total_words = 0

            for gen in item["pred"]:
                words = gen.split()
                total_words += len(words)
                unigrams.update(words)
                for i in range(len(words) - 1):
                    bigrams.add(f"{words[i]}_{words[i+1]}")
                for i in range(len(words) - 2):
                    trigrams.add(f"{words[i]}_{words[i+1]}_{words[i+2]}")

            if total_words > 0:
                dist1.append(len(unigrams) / total_words)
                dist2.append(len(bigrams)  / total_words)
                dist3.append(len(trigrams) / total_words)
            else:
                dist1.append(0.0)
                dist2.append(0.0)
                dist3.append(0.0)
                warnings.warn("[WARNING]: total_words is 0, dist-n unreliable")

        return (float(np.nanmean(dist1)),
                float(np.nanmean(dist2)),
                float(np.nanmean(dist3)))

    def _evaluate_toxigen(self, results: List[Dict]) -> Dict:
        """
        Score generated texts with tomh/toxigen_roberta.
        Change 3: model loaded from HuggingFace instead of local path.
        Change 4: uses continuous _score_probabilities instead of binary
                  _score_generations for finer resolution.
        Returns toxigen_overall in [0, 1] where 1 = fully safe.
        """
        toxigen_tokenizer  = AutoTokenizer.from_pretrained(TOXIGEN_MODEL_ID)
        toxigen_classifier = AutoModelForSequenceClassification.from_pretrained(
            TOXIGEN_MODEL_ID).to(self.device)

        preds = []
        for item in results:
            preds.extend(item["pred"])

        # Change 4: continuous probability scoring
        probabilities = self._score_probabilities(
            toxigen_tokenizer, toxigen_classifier, preds, batch_size=64
        )

        del toxigen_classifier
        torch.cuda.empty_cache()

        toxigen_score = 1.0 - sum(probabilities) / len(probabilities)
        return {"toxigen_overall": toxigen_score}

    @torch.no_grad()
    def _score_probabilities(
        self,
        tokenizer,
        model,
        preds: List[str],
        batch_size: int = 64,
    ) -> List[float]:
        """
        Change 5 (new method): continuous P(toxic) at class index 1.
        Decorated with @torch.no_grad() — was missing in the modified version.
        """
        probabilities = []
        for i in range(0, len(preds), batch_size):
            batch  = preds[i : i + batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(model.device)
            outputs     = model(**inputs)
            toxic_probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            probabilities.extend(toxic_probs.tolist())
        return probabilities

    # ── N-gram fluency helpers ───────────────────────────────────────────────

    def _n_gram_entropy(self, gen_texts, agg="arith"):
        assert agg in ("arith", "geom")
        fn = scipy.stats.mstats.gmean if agg == "geom" else np.mean
        return fn([self._compute_n_gram_entropy(t) for t in gen_texts]).item()

    def _compute_n_gram_entropy(self, sentence, ns=None, weights=None, agg="arith"):
        if ns      is None: ns      = [2, 3]
        if weights is None: weights = [2/3, 4/3]
        assert agg in ("arith", "geom")

        entropy_list = []
        for n in ns:
            tokens = nltk.word_tokenize(sentence)
            ngrams = nltk.ngrams(tokens, n)
            fdist  = nltk.FreqDist(ngrams)
            freqs  = np.array([freq for _, freq in fdist.items()], dtype=float)
            freqs /= freqs.sum()
            entropy_list.append(float(np.sum(-freqs * np.log(freqs) / np.log(2))))

        entropy_list = np.array(entropy_list) * np.array(weights)
        fn = scipy.stats.mstats.gmean if agg == "geom" else np.mean
        return fn(entropy_list)


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CAA steering experiment results."
    )
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to write evaluation output files")
    parser.add_argument("--eval_methods", nargs="+", default=None,
                        help="Methods: ppl sentiment distinctness toxigen fluency")
    parser.add_argument("--generation_dataset_path", type=str, required=True,
                        help="Path to the generation results JSON file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="Model name/path (required for ppl evaluation)")
    parser.add_argument("--save_results", type=bool, default=True,
                        help="Whether to save evaluation results to disk")
    args = parser.parse_args()

    evaluator = Evaluator(**vars(args))
    evaluator.evaluate_all()
