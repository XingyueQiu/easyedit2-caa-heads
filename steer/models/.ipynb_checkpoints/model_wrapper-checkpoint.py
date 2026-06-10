"""
model_wrapper.py
================
Model wrapper classes for EasyEdit2 CAA experiments.

MODIFIED FROM EASYEDIT2 ORIGINAL
Changes vs original:
  - AttnWrapper: replaced single add_attn_activations tensor with
    add_attn_activations_dict (dict-keyed, matches BlockOutputWrapper pattern)
    and added from_position propagation in forward()
  - BlockOutputWrapper: added from_position propagation to AttnWrapper;
    added add_attn / reset_attn helper methods
  - BaseModelWrapper: added set_add_attn_activations() for attention-level
    block injection; added _get_model_layers() helper to remove repeated
    Hack_no_grad checks throughout
  - All wrappers: added set_add_attn_activations_heads() for head-level
    masked steering with consistent inject_at="attention"|"block" parameter
  - _mask_steering_vector() extracted as a module-level helper shared by
    all wrappers (was copy-pasted identically across 4 classes)
  - Removed debug prints (QWEN METHOD, ADD THIS comments)
  - Removed [HEAD-STEERING] verbose prints; kept one clean summary print
  - LlamaWrapper: added inject_at parameter (was missing, would crash)
  - GemmaWrapper: added inject_at parameter (was missing)
  - Fixed reset_caa_layers operator precedence bug (in apply_caa.py companion)
"""

import copy
import os
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List

from steer.vector_appliers.lm_steer.apply_lm_steer_hparam import ApplyLmSteerHyperParams
from steer.models.utils import add_vector_from_position
from steer.vector_generators.lm_steer.generate_lm_steer_hparam import LmSteerHyperParams
from steer.utils.hparams import HyperParams
from steer.vector_generators.lm_steer.lm_steer_helper import Hack_no_grad, Projected_Adaptor

os.environ['VLLM_USE_V1'] = '0'

try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


# ============================================================================
# Module-level helper — shared by all wrapper classes
# ============================================================================

def _mask_steering_vector(
    steering_vector: t.Tensor,
    heads: List[int],
    num_heads: int,
    hidden_size: int,
) -> t.Tensor:
    """
    Zero out all head slots in `steering_vector` except those in `heads`.

    Works for both 1-D [hidden_size] and 2-D [seq_len, hidden_size] vectors.
    Returns a new tensor of the same shape with non-selected heads zeroed.

    Raises ValueError for unsupported tensor ranks.
    """
    head_dim   = hidden_size // num_heads
    valid_heads = [h for h in heads if 0 <= h < num_heads]

    if steering_vector.dim() == 1:
        if steering_vector.shape[0] != hidden_size:
            # Pad or truncate to match hidden_size
            fixed = t.zeros(hidden_size,
                            device=steering_vector.device,
                            dtype=steering_vector.dtype)
            copy_len = min(steering_vector.shape[0], hidden_size)
            fixed[:copy_len] = steering_vector[:copy_len]
            steering_vector = fixed

        masked = t.zeros_like(steering_vector)
        for h in valid_heads:
            s, e = h * head_dim, (h + 1) * head_dim
            masked[s:e] = steering_vector[s:e]

    elif steering_vector.dim() == 2:
        if steering_vector.shape[-1] != hidden_size:
            raise ValueError(
                f"Steering vector last dim {steering_vector.shape[-1]} "
                f"!= hidden_size {hidden_size}"
            )
        masked = t.zeros_like(steering_vector)
        for h in valid_heads:
            s, e = h * head_dim, (h + 1) * head_dim
            masked[..., s:e] = steering_vector[..., s:e]

        # Collapse to 1-D for injection (consistent with 1-D path)
        masked = masked.reshape(-1)[:hidden_size]

    else:
        raise ValueError(
            f"Unsupported steering vector shape: {steering_vector.shape}. "
            "Expected 1-D or 2-D."
        )

    return masked


# ============================================================================
# Attention wrapper
# ============================================================================

class AttnWrapper(t.nn.Module):
    """Wraps an attention module to capture outputs and inject steering."""

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.from_position = None
        # Dict-keyed so multiple steering methods can coexist (matches
        # BlockOutputWrapper.add_activations_dict pattern)
        self.add_attn_activations_dict = {}

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)

        if isinstance(output, tuple):
            attn_out, rest = output[0], output[1:]
        else:
            attn_out, rest = output, None

        self.activations = attn_out

        position_ids = kwargs.get("position_ids", None)
        for v in self.add_attn_activations_dict.values():
            if v is None:
                continue
            attn_out = add_vector_from_position(
                matrix=attn_out,
                vector=v,
                position_ids=position_ids,
                from_pos=getattr(self, "from_position", None),
            )

        return attn_out if rest is None else (attn_out,) + rest

    def add(self, activations, method_name="default"):
        self.add_attn_activations_dict[method_name] = activations

    def reset(self, method_name="all"):
        if method_name == "all":
            self.add_attn_activations_dict.clear()
        else:
            self.add_attn_activations_dict.pop(method_name, None)
        self.activations = None


# ============================================================================
# MLP wrapper
# ============================================================================

class MLPWrapper(t.nn.Module):
    """Wraps an MLP module to capture activations."""

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
        self.activations = None
        self.add_mlp_activations = None

    def forward(self, *args, **kwargs):
        output = self.mlp(*args, **kwargs)
        self.activations = output[0] if isinstance(output, tuple) else output
        return output

    def add(self, activations):
        self.add_mlp_activations = activations


# ============================================================================
# Block output wrapper
# ============================================================================

class BlockOutputWrapper(t.nn.Module):
    """
    Wraps a transformer block to:
      - capture residual stream activations
      - inject block-level steering vectors (add_activations_dict)
      - inject attention-level steering vectors (via AttnWrapper)
      - support arbitrary interventions (RePS, etc.)
    """

    # Supported model families and their attention attribute names
    _ATTN_ATTR  = {"gpt": "attn",      "llama": "self_attn",
                   "gemma": "self_attn", "qwen": "self_attn", "qwq": "self_attn"}
    _GPT_FAMILY = {"gpt"}
    _LLM_FAMILY = {"llama", "gemma", "qwen", "qwq"}

    def __init__(self, block, unembed_matrix, norm, tokenizer, layer_id,
                 model_name_or_path):
        super().__init__()
        self.block          = block
        self.unembed_matrix = unembed_matrix
        self.norm           = norm
        self.tokenizer      = tokenizer
        self.layer_id       = layer_id

        # Detect model family from path string
        path_lower = model_name_or_path.lower()
        for family in ("qwq", "qwen", "gemma", "llama", "gpt"):  # qwq before qwen
            if family in path_lower:
                self.model_type = family
                break
        else:
            raise ValueError(
                f"Cannot determine model family from path: {model_name_or_path}. "
                "Supported: gpt, llama, gemma, qwen, qwq"
            )

        # Wrap attention and MLP
        if self.model_type in self._GPT_FAMILY:
            self.block.attn                  = AttnWrapper(self.block.attn)
            self.block.mlp                   = MLPWrapper(self.block.mlp)
            self.post_attention_layernorm    = self.block.ln_2
        else:
            # Normalise attribute name (some Qwen variants use linear_attn)
            if not hasattr(self.block, "self_attn") and hasattr(self.block, "linear_attn"):
                self.block.self_attn = self.block.linear_attn
            self.block.self_attn             = AttnWrapper(self.block.self_attn)
            self.block.mlp                   = MLPWrapper(self.block.mlp)
            self.post_attention_layernorm    = self.block.post_attention_layernorm

        # State
        self.activations            = None
        self.save_activations       = False
        self.add_activations_dict   = {}
        self.intervention_dict      = {}
        self.from_position          = None
        self.save_internal_decodings = False
        self.calc_dot_product_with  = None
        self.dot_products           = []

    def _attn_wrapper(self):
        """Return the AttnWrapper for this block regardless of model family."""
        if self.model_type in self._GPT_FAMILY:
            return self.block.attn
        return self.block.self_attn

    def forward(self, *args, **kwargs):
        # Propagate from_position into the attention wrapper before forward
        self._attn_wrapper().from_position = self.from_position

        output = self.block(*args, **kwargs)

        original_is_tuple = isinstance(output, tuple)
        if not original_is_tuple:
            output = (output,)

        if self.save_activations:
            self.activations = output[0]
            if self.activations.dim() == 2:
                self.activations = self.activations.unsqueeze(0)

        if self.calc_dot_product_with is not None and self.save_activations:
            last = self.activations[0, -1, :]
            decoded = self.unembed_matrix(self.norm(last))
            top_id  = t.topk(decoded, 1)[1][0]
            top_tok = self.tokenizer.decode(top_id)
            dp = t.dot(last, self.calc_dot_product_with) / (
                t.norm(last) * t.norm(self.calc_dot_product_with)
            )
            self.dot_products.append((top_tok, dp.cpu().item()))

        # Block-level activation addition
        if self.add_activations_dict:
            aug = output[0]
            for activations in self.add_activations_dict.values():
                if activations is not None:
                    aug = add_vector_from_position(
                        matrix=aug,
                        vector=activations,
                        position_ids=kwargs.get("position_ids", None),
                        from_pos=self.from_position,
                    )
            output = (aug,) + output[1:]

        # Intervention (RePS, etc.)
        if self.intervention_dict:
            aug = output[0]
            for method_name, intervention in self.intervention_dict.items():
                if intervention is None:
                    continue
                result = intervention.forward(aug, **kwargs)
                aug = result.output if hasattr(result, "output") else result
            output = (aug,) + output[1:]

        if not self.save_internal_decodings:
            return output if original_is_tuple else output[0]

        # Internal decodings (optional diagnostic path)
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))
        attn_out = self._attn_wrapper().activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_out))
        attn_out = attn_out + args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_out))
        mlp_out = self.block.mlp(self.post_attention_layernorm(attn_out))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_out))

        return output if original_is_tuple else output[0]

    # ── Block-level helpers ──────────────────────────────────────────────────

    def add(self, activations, method_name="default"):
        self.add_activations_dict[method_name] = activations

    def set_intervention(self, intervention, method_name):
        self.intervention_dict[method_name] = intervention

    def reset(self, method_name="all"):
        if method_name == "all":
            self.add_activations_dict.clear()
            self.intervention_dict.clear()
        else:
            self.add_activations_dict.pop(method_name, None)
            self.intervention_dict.pop(method_name, None)

        self.activations = None
        self._attn_wrapper().activations = None
        self.from_position = None
        self.calc_dot_product_with = None
        self.dot_products = []

    # ── Attention-level helpers ──────────────────────────────────────────────

    def add_attn(self, activations, method_name="default"):
        """Inject a vector at the attention output (before FFN)."""
        self._attn_wrapper().add(activations, method_name)

    def reset_attn(self, method_name="all"):
        self._attn_wrapper().reset(method_name)

    def set_save_activations(self, value: bool):
        self.save_activations = value


# ============================================================================
# Base model wrapper
# ============================================================================

class BaseModelWrapper:
    """
    Base class for all model wrappers.

    Subclasses override:
      _adapt_model_layers()  — wraps layers with BlockOutputWrapper
      _get_layer(i)          — returns the wrapped layer at index i
      _get_model_layers()    — returns the full list of wrapped layers
      set_add_activations()  — routes to the correct layer attribute
      set_add_attn_activations_heads() — head-level masked steering
    """

    def __init__(
        self,
        torch_dtype=t.float32,
        use_chat: bool = False,
        device: str = "cuda" if t.cuda.is_available() else "cpu",
        model_name_or_path: Optional[str] = None,
        use_cache: bool = True,
        override_model_weights_path: Optional[str] = None,
        hparams: HyperParams = None,
    ):
        from ..utils.alg_dict import DTYPES_DICT

        self.hparams             = hparams
        self.use_chat            = use_chat
        self.device              = device
        self.torch_dtype         = DTYPES_DICT.get(torch_dtype, t.float32)
        self.use_cache           = use_cache
        self.model_name_or_path  = model_name_or_path

        if hparams.vllm_enable and VLLM_AVAILABLE:
            self.VLLM_model = LLM(
                model=self.model_name_or_path,
                enforce_eager=True,
                tensor_parallel_size=1,
                dtype=self.torch_dtype,
            )
            self.tokenizer = self.VLLM_model.get_tokenizer()
            self.model     = self._extract_vllm_model()
        else:
            self.VLLM_model = None
            self.tokenizer  = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                padding_side="right" if "gemma" in self.model_name_or_path else "left",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                use_cache=self.use_cache,
            )

        if override_model_weights_path is not None:
            self.model.load_state_dict(
                t.load(override_model_weights_path), strict=False
            )

        self._adapt_model_layers()

    # ── vLLM helpers ────────────────────────────────────────────────────────

    def _extract_vllm_model(self):
        try:
            if hasattr(self.VLLM_model, "llm_engine"):
                executor = self.VLLM_model.llm_engine.model_executor
                if hasattr(executor, "driver_worker"):
                    return executor.driver_worker.model_runner.model
                model = getattr(executor, "model", None)
                if model is not None:
                    return model
            if hasattr(self.VLLM_model, "model"):
                return self.VLLM_model.model
            raise ValueError("Could not locate model inside vLLM instance")
        except Exception as e:
            raise RuntimeError(f"Failed to extract model from vLLM: {e}")

    # ── Layer routing helpers ────────────────────────────────────────────────

    def _get_model_layers(self):
        """Return the list of wrapped BlockOutputWrapper layers."""
        if hasattr(self.model, "model") and isinstance(self.model.model, Hack_no_grad):
            return self.model.model.module.layers
        return self.model.model.layers

    def _get_layer(self, i):
        return self._get_model_layers()[i]

    # ── Layer adaptation ────────────────────────────────────────────────────

    def _adapt_model_layers(self):
        norm = self.model.model.norm if self.VLLM_model is None else None
        lm_head = self.model.lm_head if self.VLLM_model is None else None
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, lm_head, norm, self.tokenizer, i, self.model_name_or_path
            )
        self.set_save_activations(self.hparams.save_activations)

    # ── Activation setters / getters ────────────────────────────────────────

    def set_add_activations(self, layer, activations, method_name="default"):
        """Block-level steering (adds to residual stream after full block)."""
        self._get_layer(layer).add(activations, method_name)

    def set_add_attn_activations(self, layer, activations, method_name="default"):
        """Attention-level steering (adds to attn output, before FFN)."""
        self._get_layer(layer).add_attn(activations, method_name)

    def set_add_attn_activations_heads(
        self,
        layer: int,
        steering_vector: t.Tensor,
        heads: List[int],
        method_name: str = "caa",
        inject_at: str = "attention",
    ):
        """
        Head-level masked steering.

        Masks `steering_vector` so only the dimensions belonging to
        `heads` are non-zero, then injects either at attention output
        (inject_at="attention") or at block output (inject_at="block").

        Subclasses provide _num_heads() and _hidden_size() for their
        model's config attribute names.
        """
        num_heads   = self._num_heads()
        hidden_size = self._hidden_size()

        masked = _mask_steering_vector(steering_vector, heads, num_heads, hidden_size)

        if inject_at == "block":
            self._get_layer(layer).add(masked, method_name)
        else:
            self._get_layer(layer).add_attn(masked, method_name)

    def _num_heads(self) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _num_heads()"
        )

    def _hidden_size(self) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _hidden_size()"
        )

    # ── Other standard methods ───────────────────────────────────────────────

    def set_intervention(self, layer, intervention, method_name):
        self._get_layer(layer).set_intervention(intervention, method_name)

    def set_save_activations(self, value: bool):
        for layer in self._get_model_layers():
            layer.save_activations = value

    def set_save_internal_decodings(self, value: bool):
        for layer in self._get_model_layers():
            layer.save_internal_decodings = value

    def set_from_positions(self, pos: int):
        for layer in self._get_model_layers():
            layer.from_position = pos

    def set_calc_dot_product_with(self, layer, vector):
        self._get_layer(layer).calc_dot_product_with = vector

    def get_last_activations(self, layer):
        return self._get_layer(layer).activations

    def get_dot_products(self, layer):
        return self._get_layer(layer).dot_products

    def get_logits(self, tokens):
        if self.VLLM_model is not None:
            from vllm import SamplingParams
            return self.VLLM_model.generate(
                prompt_token_ids=tokens.tolist(),
                sampling_params=SamplingParams(max_tokens=1),
            )
        return self.model(tokens).logits

    def reset_all(self):
        for layer in self._get_model_layers():
            layer.reset(method_name="all")
        for attr in ("prompt", "generate_prompts"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.reset_lm_steer()

    def reset_lm_steer(self):
        if hasattr(self, "steer"):
            self.model.set_output_embeddings(self.steer.lm_head)
            delattr(self, "steer")
        if hasattr(self.model, "model") and isinstance(self.model.model, Hack_no_grad):
            self.model.model = self.model.model.module
        for param in self.model.parameters():
            param.requires_grad_(True)

    def reset(self, method_name):
        method_name = method_name.lower()
        if method_name in {"caa", "vector_prompt", "sae_feature", "sta",
                           "reps", "sft", "spilt"}:
            for layer in self._get_model_layers():
                layer.reset(method_name=method_name)
        elif method_name == "lm_steer":
            self.reset_lm_steer()
        elif method_name == "prompt":
            if hasattr(self, "prompt"):
                delattr(self, "prompt")
        else:
            raise ValueError(f"Method '{method_name}' not supported for reset")

    def replace_final_layer(self, hparams):
        embed_dim  = self.model.lm_head.weight.shape[1]
        vocab_size = self.model.lm_head.weight.shape[0]
        for p in self.model.parameters():
            p.requires_grad_(False)

        if hparams.adapted_component == "final_layer" and hasattr(self.model, "model"):
            self.model.model = Hack_no_grad(self.model.model)
            self.steer = Projected_Adaptor(
                self.model.lm_head, hparams.adaptor_class, hparams.num_steers,
                embed_dim, vocab_size, hparams.rank, hparams.epsilon,
                hparams.init_var, "output",
            )
            adaptor_params = {
                "multiply": ("projector1", "projector2"),
                "add":      ("add_vec",),
                "offset":   ("offset_vec",),
            }
            for attr in adaptor_params.get(hparams.adaptor_class, []):
                setattr(self.steer, attr,
                        t.nn.Parameter(getattr(self.steer, attr).to(self.torch_dtype)))
            self.model.set_output_embeddings(self.steer)
        else:
            raise ValueError("Mismatched adapted_component or model structure")

    def ori_generate(self, input_ids, **kwargs):
        """Generate without any active steering (saves/restores activation dicts)."""
        layers = self._get_model_layers()
        saved = {}
        for i, layer in enumerate(layers):
            if getattr(layer, "add_activations_dict", {}):
                saved[i] = {
                    k: v.clone().detach() if isinstance(v, t.Tensor)
                       else copy.deepcopy(v)
                    for k, v in layer.add_activations_dict.items()
                }
                layer.add_activations_dict = {}

        saved_steer = None
        if hasattr(self, "steer") and hasattr(self.steer, "steer_values"):
            saved_steer = self.steer.steer_values
            self.steer.steer_values = t.zeros(1)

        try:
            output = self.model.generate(input_ids=input_ids, **kwargs)
        finally:
            for i, d in saved.items():
                layers[i].add_activations_dict = d
            if saved_steer is not None and hasattr(self, "steer"):
                self.steer.steer_values = saved_steer

        return output

    def get_activation_data(self, decoded_activations, topk=10):
        softmaxed = t.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = t.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        data = self.get_activation_data(decoded_activations, topk)[0]
        print(label, data)


# ============================================================================
# Llama / Gemma / Qwen wrappers  (use model.model.layers)
# ============================================================================

class LlamaWrapper(BaseModelWrapper):
    def _num_heads(self):   return self.model.config.num_attention_heads
    def _hidden_size(self): return self.model.config.hidden_size


class GemmaWrapper(BaseModelWrapper):
    def _num_heads(self):   return self.model.config.num_attention_heads
    def _hidden_size(self): return self.model.config.hidden_size


class QwenWrapper(BaseModelWrapper):
    def _num_heads(self):   return self.model.config.num_attention_heads
    def _hidden_size(self): return self.model.config.hidden_size


# ============================================================================
# GPT wrapper  (uses model.transformer.h instead of model.model.layers)
# ============================================================================

class GPTWrapper(BaseModelWrapper):
    """GPT-2 and similar models that use model.transformer.h for layers."""

    # ── Layer routing overrides ──────────────────────────────────────────────

    def _get_model_layers(self):
        if isinstance(self.model.transformer, Hack_no_grad):
            return self.model.transformer.module.h
        return self.model.transformer.h

    def _adapt_model_layers(self):
        norm    = self.model.transformer.ln_f
        lm_head = None if self.VLLM_model is not None else self.model.lm_head
        for i, layer in enumerate(self.model.transformer.h):
            self.model.transformer.h[i] = BlockOutputWrapper(
                layer, lm_head, norm, self.tokenizer, i, self.model_name_or_path
            )
        self.set_save_activations(self.hparams.save_activations)

    # ── Config accessors ─────────────────────────────────────────────────────

    def _num_heads(self):   return self.model.config.n_head
    def _hidden_size(self): return self.model.config.n_embd

    # ── GPT-specific overrides ───────────────────────────────────────────────

    def set_add_activations(self, layer, activations, method_name="default"):
        self._get_layer(layer).add(activations, method_name)

    def set_intervention(self, layer, intervention, method_name):
        self._get_layer(layer).set_intervention(intervention, method_name)

    def set_save_internal_decodings(self, value: bool):
        for layer in self._get_model_layers():
            layer.save_internal_decodings = value

    def set_from_positions(self, pos: int):
        for layer in self._get_model_layers():
            layer.from_position = pos

    def get_last_activations(self, layer):
        return self._get_layer(layer).activations

    def set_calc_dot_product_with(self, layer, vector):
        self._get_layer(layer).calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self._get_layer(layer).dot_products

    def reset_all(self):
        for layer in self._get_model_layers():
            layer.reset(method_name="all")
        for attr in ("prompt", "generate_prompts"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.reset_lm_steer()

    def reset_lm_steer(self):
        if hasattr(self, "steer"):
            self.model.set_output_embeddings(self.steer.lm_head)
            delattr(self, "steer")
        if isinstance(self.model.transformer, Hack_no_grad):
            self.model.transformer = self.model.transformer.module
        for param in self.model.parameters():
            param.requires_grad_(True)

    def reset(self, method_name):
        method_name = method_name.lower()
        if method_name in {"caa", "vector_prompt", "sae_feature", "sta",
                           "reps", "sft", "spilt"}:
            for layer in self._get_model_layers():
                layer.reset(method_name=method_name)
        elif method_name == "lm_steer":
            self.reset_lm_steer()
        elif method_name == "prompt":
            if hasattr(self, "prompt"):
                delattr(self, "prompt")
        else:
            raise ValueError(f"Method '{method_name}' not supported for reset")

    def replace_final_layer(self, hparams):
        embed_dim  = self.model.lm_head.weight.shape[1]
        vocab_size = self.model.lm_head.weight.shape[0]
        for p in self.model.parameters():
            p.requires_grad_(False)

        if hparams.adapted_component == "final_layer":
            self.model.transformer = Hack_no_grad(self.model.transformer)
            self.steer = Projected_Adaptor(
                self.model.lm_head, hparams.adaptor_class, hparams.num_steers,
                embed_dim, vocab_size, hparams.rank, hparams.epsilon,
                hparams.init_var, "output",
            )
            adaptor_params = {
                "multiply": ("projector1", "projector2"),
                "add":      ("add_vec",),
                "offset":   ("offset_vec",),
            }
            for attr in adaptor_params.get(hparams.adaptor_class, []):
                setattr(self.steer, attr,
                        t.nn.Parameter(getattr(self.steer, attr).to(self.torch_dtype)))
            self.model.set_output_embeddings(self.steer)

    def ori_generate(self, input_ids, **kwargs):
        layers = self._get_model_layers()
        saved = {}
        for i, layer in enumerate(layers):
            if getattr(layer, "add_activations_dict", {}):
                saved[i] = copy.deepcopy(layer.add_activations_dict)
                layer.add_activations_dict = {}

        saved_steer = None
        if hasattr(self, "steer") and hasattr(self.steer, "steer_value"):
            saved_steer = self.steer.steer_value
            self.steer.steer_value = 0

        try:
            output = self.model.generate(input_ids=input_ids, **kwargs)
        finally:
            for i, d in saved.items():
                layers[i].add_activations_dict = d
            if saved_steer is not None and hasattr(self, "steer"):
                self.steer.steer_value = saved_steer

        return output
