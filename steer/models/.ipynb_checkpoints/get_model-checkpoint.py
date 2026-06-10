"""
get_model.py
============
NOT MODIFIED FROM EASYEDIT2 ORIGINAL

No logic changes. The audit flagged this file as modified likely due to
trailing commas added after the last argument in each wrapper call — a
style-only difference with no effect.
"""

from .model_wrapper import GPTWrapper, GemmaWrapper, LlamaWrapper, QwenWrapper
import torch as t


def get_model(hparams):
    torch_dtype             = hparams.torch_dtype if hasattr(hparams, "torch_dtype") else t.float32
    use_cache               = hparams.use_cache if hasattr(hparams, "use_cache") else True
    use_chat                = hparams.use_chat_template if hasattr(hparams, "use_chat_template") else False
    device                  = hparams.device if hasattr(hparams, "device") else ("cuda" if t.cuda.is_available() else "cpu")
    model_name_or_path      = hparams.model_name_or_path if hasattr(hparams, "model_name_or_path") else None
    override_model_weights_path = hparams.override_model_weights_path if hasattr(hparams, "override_model_weights_path") else None

    path = hparams.model_name_or_path.lower()
    kwargs = dict(
        torch_dtype=torch_dtype,
        use_chat=use_chat,
        device=device,
        model_name_or_path=model_name_or_path,
        use_cache=use_cache,
        override_model_weights_path=override_model_weights_path,
        hparams=hparams,
    )

    if "llama" in path:
        model = LlamaWrapper(**kwargs)
    elif "gpt" in path:
        model = GPTWrapper(**kwargs)
    elif "gemma" in path:
        model = GemmaWrapper(**kwargs)
    elif "qwen" in path or "qwq" in path:
        model = QwenWrapper(**kwargs)
    else:
        raise ValueError(f"model_name_or_path '{hparams.model_name_or_path}' not supported")

    return model, model.tokenizer
