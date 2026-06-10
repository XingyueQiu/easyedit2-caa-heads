"""
lm_steer_data.py
================
MODIFIED FROM EASYEDIT2 ORIGINAL

Change 1  load_contrastive_data: added question-prepending support —
          if the dataset has a "question" field, the question text is
          prepended to each matching/not_matching response before
          building the model input. Original had no question handling.

Change 2  Added multimodal support functions (all new, not in original):
            load_multimodal_contrastive_data
            load_multimodal_labed_data
            load_multimodal_lm_steer_dataset
          These handle image+text datasets via a processor instead of a
          tokenizer. Not used by your toxicity/SST2 experiment.

Note: Change 1 may affect your experiment if your dataset has a "question"
      field — the question will be prepended to the answer text during
      LM-Steer vector training. Verify this is the intended behaviour for
      your dataset format.
"""

from ..utils import build_model_input, build_multimodal_model_input
from PIL import Image


def load_contrastive_data(train_data, subset, tokenizer,
                          system_prompt=None, use_chat_template=False):
    pos_data = [{"text": item["matching"],     "label":  1} for item in train_data]
    neg_data = [{"text": item["not_matching"], "label": -1} for item in train_data]

    # Change 1: prepend question text if present in dataset
    if "question" in train_data[0]:
        ques_data = [{"text": item.get("question", "")} for item in train_data]
        ques_data = ques_data + ques_data  # mirrors pos + neg order

    if subset is not None:
        pos_data = pos_data[:subset]
        neg_data = neg_data[:subset]

    dataset = pos_data + neg_data
    for i, _datum in enumerate(dataset):
        if not isinstance(_datum["text"], str):
            _datum["text"] = str(_datum["text"])
        if "question" in train_data[0]:  # Change 1
            _datum["text"] = ques_data[i]["text"] + _datum["text"]
        _datum["text"] = build_model_input(
            _datum["text"], tokenizer, system_prompt, use_chat_template
        )
    return dataset


def load_labed_data(train_data, tokenizer,
                    system_prompt=None, use_chat_template=False):
    labels = [int(item["label"]) for item in train_data]
    min_label, max_label = min(labels), max(labels)
    dataset = []
    for item in train_data:
        mapped_label = (int(item["label"]) - min_label) / (max_label - min_label) * 2 - 1
        if not isinstance(item["text"], str):
            item["text"] = str(item["text"])
        item["text"] = build_model_input(
            item["text"], tokenizer, system_prompt, use_chat_template
        )
        dataset.append({"text": item["text"], "label": mapped_label})
    return dataset


def load_lm_steer_dataset(raw_data, subset, tokenizer, system_prompt, use_chat_template):
    if "text" in raw_data[0] and "label" in raw_data[0]:
        return load_labed_data(raw_data, tokenizer, system_prompt, use_chat_template)
    elif "matching" in raw_data[0] and "not_matching" in raw_data[0]:
        return load_contrastive_data(raw_data, subset, tokenizer, system_prompt, use_chat_template)
    else:
        raise NotImplementedError()


# ── Change 2: multimodal support (new, not in original) ──────────────────────

def load_multimodal_contrastive_data(train_data, subset, processor,
                                      system_prompt=None, use_chat_template=False):
    def _open_image(img):
        return img if isinstance(img, Image.Image) else Image.open(img)

    has_image    = "image"    in train_data[0]
    has_question = "question" in train_data[0]

    if has_image:
        pos_data = [{"text": item["matching"],     "image": _open_image(item["image"]), "label":  1} for item in train_data]
        neg_data = [{"text": item["not_matching"], "image": _open_image(item["image"]), "label": -1} for item in train_data]
    else:
        pos_data = [{"text": item["matching"],     "label":  1} for item in train_data]
        neg_data = [{"text": item["not_matching"], "label": -1} for item in train_data]

    if has_question:
        ques_data = [{"text": item.get("question", "")} for item in train_data] * 2

    if subset is not None:
        pos_data = pos_data[:subset]
        neg_data = neg_data[:subset]

    dataset = pos_data + neg_data
    for i, _datum in enumerate(dataset):
        if not isinstance(_datum["text"], str):
            _datum["text"] = str(_datum["text"])
        if has_question:
            if has_image:
                conversation = [
                    {"role": "user",      "content": [{"type": "text",  "text": ques_data[i]["text"]},
                                                       {"type": "image"}]},
                    {"role": "assistant", "content": _datum["text"]},
                ]
            else:
                conversation = [
                    {"role": "user",      "content": [{"type": "text", "text": ques_data[i]["text"]}]},
                    {"role": "assistant", "content": _datum["text"]},
                ]
        else:
            conversation = [{"role": "assistant", "content": _datum["text"]}]
        _datum["text"] = build_multimodal_model_input(
            conversation, processor, system_prompt, use_chat_template
        )
    return dataset


def load_multimodal_labed_data(train_data, processor,
                                system_prompt=None, use_chat_template=False):
    labels = [int(item["label"]) for item in train_data]
    min_label, max_label = min(labels), max(labels)
    has_image = "image" in train_data[0] if train_data else False

    if has_image:
        def _open_image(img):
            return img if isinstance(img, Image.Image) else Image.open(img)
        train_data = [{"text": item["text"], "image": _open_image(item["image"]),
                       "label": item["label"]} for item in train_data]

    dataset = []
    for item in train_data:
        mapped_label = (int(item["label"]) - min_label) / (max_label - min_label) * 2 - 1
        if not isinstance(item["text"], str):
            item["text"] = str(item["text"])
        if has_image:
            conversation = [
                {"role": "user",      "content": [{"type": "text",  "text": item["text"]},
                                                   {"type": "image"}]},
                {"role": "assistant", "content": ""},
            ]
        else:
            conversation = [{"role": "assistant", "content": item["text"]}]
        processed_text = build_multimodal_model_input(
            conversation, processor, system_prompt, use_chat_template
        )
        entry = {"text": processed_text, "label": mapped_label}
        if has_image:
            entry["image"] = item["image"]
        dataset.append(entry)
    return dataset


def load_multimodal_lm_steer_dataset(raw_data, subset, processor,
                                      system_prompt, use_chat_template):
    if "text" in raw_data[0] and "label" in raw_data[0]:
        return load_multimodal_labed_data(raw_data, processor, system_prompt, use_chat_template)
    elif "matching" in raw_data[0] and "not_matching" in raw_data[0]:
        return load_multimodal_contrastive_data(raw_data, subset, processor, system_prompt, use_chat_template)
    else:
        raise NotImplementedError()
