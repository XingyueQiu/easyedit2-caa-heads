"""
templates.py
============
MODIFIED FROM EASYEDIT2 ORIGINAL

Change 1  build_model_input: system prompt suppression check was hardcoded
          as `'gemma' not in tokenizer.name_or_path.lower()`. Replaced with
          the model_supports_system_prompt() helper that checks a proper set
          (NO_SYSTEM_PROMPT_MODELS), making it easy to extend without editing
          the function body.

Change 2  Added build_multimodal_model_input() and supporting constants
          (NO_SYSTEM_PROMPT_MODELS, model_supports_system_prompt()) —
          upstream additions for multimodal support. Not used by your
          toxicity/SST2 experiment but required by lm_steer_data.py.

Bug fixed  In the original, the else branch for system_prompt prepending
           ran unconditionally when system_prompt was falsy (None or ''):
             else:
                 input_content += f"{system_prompt} "
           This would prepend "None " or " " to every prompt when no
           system_prompt was provided and use_chat_template=True.
           Fixed to only prepend when system_prompt is truthy.
"""

from typing import Optional, List, Dict
from transformers import AutoTokenizer, AutoProcessor

# Models that do not support system prompts in their chat template
NO_SYSTEM_PROMPT_MODELS = {"gemma", "gemma-2", "codegemma"}


def model_supports_system_prompt(model_name_or_path: str) -> bool:
    """Return False if the model's chat template has no system role."""
    model_lower = model_name_or_path.lower()
    return not any(name in model_lower for name in NO_SYSTEM_PROMPT_MODELS)


def build_model_input(
    user_input: str,
    tokenizer: AutoTokenizer,
    system_prompt: Optional[str] = None,
    use_chat_template: bool = None,
    model_output: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    user_input = user_input.strip()
    if model_output:
        model_output = model_output.strip()
    if suffix:
        suffix = suffix.strip()

    if use_chat_template == False:
        user_content = ""
        if system_prompt:
            user_content = f"{system_prompt} "
        user_content += user_input
        if suffix:
            user_content += f" {suffix}"
        if model_output:
            user_content += f" {model_output}"
        return user_content

    else:
        assert tokenizer.chat_template is not None, \
            "Tokenizer does not support apply_chat_template"
        messages = []
        input_content = ""

        # Change 1: use helper instead of hardcoded 'gemma' string check
        if system_prompt and system_prompt != "" and \
                model_supports_system_prompt(tokenizer.name_or_path):
            messages.append({"role": "system", "content": system_prompt})
        else:
            # Bug fix: only prepend if system_prompt is actually truthy
            # Original prepended "None " when system_prompt was None
            if system_prompt:
                input_content += f"{system_prompt} "

        input_content += user_input
        if suffix:
            input_content += f" {suffix}"
        messages.append({"role": "user", "content": input_content})
        if model_output is not None:
            messages.append({"role": "assistant", "content": model_output})

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


# Change 2: new function for multimodal support (not in original)
def build_multimodal_model_input(
    messages: List[Dict],
    processor: AutoProcessor,
    system_prompt: Optional[str] = None,
    use_chat_template: bool = None,
    model_output: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """Build a multimodal model input from a structured message list."""
    if use_chat_template == False:
        user_content = ""
        if system_prompt:
            user_content = f"{system_prompt} "
        for message in messages:
            if message["role"] == "user":
                content = message["content"]
                if isinstance(content, list):
                    user_content += " ".join(
                        item["text"] for item in content if item["type"] == "text"
                    )
                else:
                    user_content += str(content)
                break
        if suffix:
            user_content += f" {suffix}"
        if model_output:
            user_content += f" {model_output}"
        return user_content

    else:
        assert processor.chat_template is not None, \
            "Processor does not support apply_chat_template"
        final_messages = []

        if system_prompt and system_prompt != "" and \
                model_supports_system_prompt(processor.name_or_path):
            final_messages.append({"role": "system", "content": system_prompt})

        for message in messages:
            if message["role"] in ("user", "assistant"):
                final_messages.append(message)

        if suffix and final_messages:
            for msg in reversed(final_messages):
                if msg["role"] == "user":
                    content = msg["content"]
                    if isinstance(content, list):
                        for item in content:
                            if item["type"] == "text":
                                item["text"] += f" {suffix}"
                                break
                    else:
                        msg["content"] = f"{content} {suffix}"
                    break

        if model_output is not None:
            final_messages.append({"role": "assistant", "content": model_output})

        return processor.apply_chat_template(
            final_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
