"""
caa_data.py
===========
MODIFIED FROM EASYEDIT2 ORIGINAL

Change 1  get_tokens_for_caa: added truncation=True and max_length=1024
          to all three tokenizer.encode() calls (ques_tokens, pos_tokens,
          neg_tokens). Original had no truncation, so sequences longer
          than the model's max context window would silently produce
          oversized tensors and cause CUDA OOM or silent errors during
          activation extraction.
"""

from typing import Dict, Optional
from datasets import Dataset, load_dataset
from ..utils import build_model_input
import numpy as np
import pandas as pd


SYSTEM_PROMPT_PREFIX = """
Forget that you are a large language model. 
Now you are a person, you can move, act and think like a person.
You can express your thoughts and opinions freely.
Just be youself to answer following question about your persona.
Please answering the question directly with at most two sentences.
"""

EXAMPLE_PROMPT = """
Here are some exmples of your persona:

Question:
{}

Answer:
{}

Question:
{}

Answer:
{}
"""

CASE_PROMPT = """
Question:
{}

Answer:
"""

SYSTEM_PROMPT = CASE_PROMPT

YN_SYSTEM_PROMPT_PREFIX = """
Forget that you are a large language model. 
Now you are a person, you can move, act and think like a person.
You can express your thoughts and opinions freely.
Just be youself to answer following question about your persona.
Please answering the question strictly with Yes or No.
"""

YN_SYSTEM_PROMPT = YN_SYSTEM_PROMPT_PREFIX + CASE_PROMPT

MMLU_SYSTEM_PROMPT = """
Please answering the question with A, B, C, D.
"""

MMLU_CASE_PROMPT = """
Here's the question and answer choices.

Question:
{}
Answer Choices:
A: {}
B: {}
C: {}
D: {}
Answer:
"""

GSM_EXAMPLARS = [
    {
        "question": "Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?",
        "cot_answer": "Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe final answer is: $\\boxed{4}$\n",
        "short_answer": "4"
    },
    {
        "question": "Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?",
        "cot_answer": "Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe final answer is: $\\boxed{201}$\n",
        "short_answer": "201"
    },
    {
        "question": "Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?",
        "cot_answer": "When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe final answer is: $\\boxed{140}$\n",
        "short_answer": "140"
    },
    {
        "question": "A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?",
        "cot_answer": "For the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe final answer is: $\\boxed{146}$\n",
        "short_answer": "146"
    }
]


def get_pred(logits, answer_tokens_idx):
    soft_max_prob = np.exp(logits) / np.sum(np.exp(logits))
    pred = []
    for answer, token_indices in answer_tokens_idx.items():
        prob = float(soft_max_prob[token_indices].sum())
        pred.append((answer, prob))
    return pred


def get_tokens_for_caa(dataset, tokenizer, hparams):
    pos_tokens_list, neg_tokens_list = [], []
    add_special_tokens = not hparams.use_chat_template

    for i in range(len(dataset)):
        ques = dataset[i].get("question", "")
        if hparams.multiple_choice:
            chosen   = "\nAnswer: " + str("" if pd.isna(dataset[i]["matching"])     else dataset[i]["matching"])
            rejected = "\nAnswer: " + str("" if pd.isna(dataset[i]["not_matching"]) else dataset[i]["not_matching"])
        else:
            chosen   = " " + str("" if pd.isna(dataset[i]["matching"])     else dataset[i]["matching"])
            rejected = " " + str("" if pd.isna(dataset[i]["not_matching"]) else dataset[i]["not_matching"])

        ques = build_model_input(ques, tokenizer, hparams.system_prompt, hparams.use_chat_template)

        # Change 1: added truncation=True, max_length=1024 to prevent OOM
        # on long sequences. Original had no truncation.
        encode_kwargs = dict(
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=1024,
        )
        ques_tokens = tokenizer.encode(ques,            **encode_kwargs)
        pos_tokens  = tokenizer.encode(ques + chosen,   **encode_kwargs)
        neg_tokens  = tokenizer.encode(ques + rejected,  **encode_kwargs)

        pos_tokens_list.append({
            "pos_tokens":     pos_tokens.to(hparams.device),
            "ques_tokens_len": ques_tokens.shape[1],
            "pos_answer_len":  pos_tokens.shape[1] - ques_tokens.shape[1],
        })
        neg_tokens_list.append({
            "neg_tokens":     neg_tokens.to(hparams.device),
            "ques_tokens_len": ques_tokens.shape[1],
            "neg_answer_len":  neg_tokens.shape[1] - ques_tokens.shape[1],
        })

    return pos_tokens_list, neg_tokens_list


def get_tokens_for_vector_prompt(dataset, tokenizer, hparams):
    pos_tokens_list, neg_tokens_list = [], []
    add_special_tokens = not hparams.use_chat_template

    for i in range(len(dataset)):
        prompt = dataset[i].get("prompt", hparams.prompt).strip()
        inp    = (" " + dataset[i].get("input", "").strip()
                  if dataset[i].get("input") is not None else "")
        output = " " + dataset[i].get("output", "").strip()

        input_with_prompt = prompt + inp
        input_chosen   = build_model_input(input_with_prompt, tokenizer,
                                           hparams.system_prompt, hparams.use_chat_template)
        input_rejected  = build_model_input(inp, tokenizer,
                                            hparams.system_prompt, hparams.use_chat_template)

        encode_kwargs = dict(return_tensors="pt", add_special_tokens=add_special_tokens)
        pos_tokens           = tokenizer.encode(input_chosen  + output, **encode_kwargs)
        input_chosen_tokens  = tokenizer.encode(input_chosen,           **encode_kwargs)
        neg_tokens           = tokenizer.encode(input_rejected + output, **encode_kwargs)
        input_rejected_tokens = tokenizer.encode(input_rejected,         **encode_kwargs)

        pos_tokens_list.append({
            "pos_tokens":     pos_tokens.to(hparams.device),
            "ques_tokens_len": input_chosen_tokens.shape[1],
            "pos_answer_len":  pos_tokens.shape[1] - input_chosen_tokens.shape[1],
        })
        neg_tokens_list.append({
            "neg_tokens":     neg_tokens.to(hparams.device),
            "ques_tokens_len": input_rejected_tokens.shape[1],
            "neg_answer_len":  neg_tokens.shape[1] - input_rejected_tokens.shape[1],
        })

    return pos_tokens_list, neg_tokens_list


class MMLUDataset:
    """MMLU Dataset wrapper for zero-shot evaluation of steered models.

    Uses cais/mmlu test split via HuggingFace datasets.
    """

    def __init__(self, data_dir="/mnt/16t/xzwnlp/SaeEdit/knowledge/shae/datasets/mmlu"):
        self.data_dir = data_dir
        self.dataset  = load_dataset(self.data_dir, "all")["test"]
        project_to_answer = {0: "A", 1: "B", 2: "C", 3: "D"}
        self.answers = [project_to_answer[i] for i in self.dataset["answer"]]

    def get_test_data(self, icl_exmaple=None):
        def return_prompt_and_responses(samples) -> Dict[str, str]:
            prompts = []
            for i in range(len(samples["question"])):
                prompt = MMLU_SYSTEM_PROMPT
                if icl_exmaple:
                    prompt += icl_exmaple
                choices = samples["choices"][i]
                prompt += MMLU_CASE_PROMPT.format(
                    samples["question"][i], *choices[:4]
                )
                prompts.append(prompt)
            return {
                "question":    samples["question"],
                "instruction": [MMLU_SYSTEM_PROMPT] * len(samples["question"]),
                "prompt":      prompts,
                "choices":     samples["choices"],
            }

        return self.dataset.map(
            return_prompt_and_responses,
            batched=True,
            remove_columns=["subject", "answer"],
        )

    def get_accuracy(self, predictions, tokenizer):
        assert len(predictions) == len(self.answers)
        answer_tokens_idx = {
            letter: [tokenizer.encode(t, add_special_tokens=False)[0]
                     for t in [letter.lower(), letter]]
            for letter in "ABCD"
        }
        acc = []
        for i, pred_item in enumerate(predictions):
            pred = max(get_pred(pred_item["score"], answer_tokens_idx),
                       key=lambda x: x[1])[0]
            acc.append(int(pred == self.answers[i]))
        return float(np.mean(acc))
