"""
Model loading utilities for Phase 1 (Feature Discovery).

Provides a thin, reproducible wrapper around transformer_lens HookedTransformer
for loading 7B-13B class instruct models on Apple Silicon (MPS) or CPU.

Primary target:
- meta-llama/Meta-Llama-3.1-8B-Instruct

Secondary:
- mistralai/Mistral-7B-Instruct-v0.3 (or current equivalent)

Usage:
    from src.utils.model_loading import load_model, get_residual_stream_layers

    model = load_model("llama-3.1-8b", device="auto")
    print(model.cfg.n_layers)
"""

from __future__ import annotations

import os
from typing import Optional, List, Dict, Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    from transformer_lens import HookedTransformer
except ImportError as e:
    HookedTransformer = None  # type: ignore


SUPPORTED_MODELS: Dict[str, str] = {
    # Primary targets - use the exact strings that this TL version recognizes
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    # Public Llama family
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
}


def get_device(device: Optional[str] = "auto") -> str:
    """Resolve the best available device.

    Priority: explicit -> mps (Apple) -> cuda -> cpu
    """
    if device is not None and device != "auto":
        return device
    if torch is None:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_dtype_for_device(device: str, prefer_bf16: bool = True):
    """Choose a reasonable dtype.

    On MPS we default to float32 for large models to avoid Metal bugs.
    """
    if torch is None:
        return None
    if device == "mps":
        return torch.float32
    if device == "cuda" and prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_model(
    model_name: str = "llama-3.1-8b",
    device: Optional[str] = "auto",
    dtype: Optional[torch.dtype] = None,
    center_writing_weights: bool = True,
    center_unembed: bool = True,
    fold_ln: bool = True,
    refactor_factored_attn_matrices: bool = False,
    **from_pretrained_kwargs: Any,
) -> HookedTransformer:
    """
    Load a model using transformer_lens.

    Args:
        model_name: Short name (see SUPPORTED_MODELS) or full HF repo id.
        device: 'auto', 'mps', 'cuda', or 'cpu'.
        dtype: Override dtype. If None, chosen based on device.
        **from_pretrained_kwargs: Passed through to HookedTransformer.from_pretrained.
            Common: hf_token=..., cache_dir=..., n_devices=1

    Returns:
        HookedTransformer ready for activation caching / SAE training.

    Notes:
        - Llama-3.1-8B-Instruct is gated on HF. Run `huggingface-cli login` first
          or pass hf_token=... .
        - For memory, start with batch_size=1-4 and short seq lens during exploration.
        - Some SAEs in sae_lens expect models with specific cfg (e.g. d_model).
    """
    if torch is None or HookedTransformer is None:
        raise ImportError(
            "torch and transformer_lens are required to load models. "
            "Run: pip3 install -r requirements.txt"
        )

    device = get_device(device)
    if dtype is None:
        dtype = get_dtype_for_device(device)

    # MPS + float16 is currently unstable for large Llama models in transformer_lens (matmul dtype errors).
    # Force float32 on MPS for safety during Phase 1 experiments.
    if device == "mps":
        os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
        if dtype == torch.float16:
            print("[model_loading] Forcing float32 on MPS to avoid known Metal matmul dtype bugs with large models.")
            dtype = torch.float32

    hf_model_id = SUPPORTED_MODELS.get(model_name.lower(), model_name)

    print(f"[model_loading] Loading {hf_model_id}")
    print(f"[model_loading] device={device}, dtype={dtype}")

    model = None
    try:
        # First attempt: direct (works for models known to this version of TL)
        model = HookedTransformer.from_pretrained(
            hf_model_id,
            device=device,
            dtype=dtype,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            fold_ln=fold_ln,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
            **from_pretrained_kwargs,
        )
    except Exception as e:
        err = str(e)
        if "not found" in err.lower() or "official model" in err.lower() or "valid official" in err.lower():
            print("[model_loading] Model not in TL official list. Using HF load + wrap fallback (recommended for Llama-3.1+).")
            from transformers import AutoModelForCausalLM
            import torch as _torch

            hf_model = AutoModelForCausalLM.from_pretrained(
                hf_model_id,
                torch_dtype=dtype or _torch.float16,
                low_cpu_mem_usage=True,
            )
            # Good defaults for modern Llama
            model = HookedTransformer.from_pretrained(
                hf_model_id,
                hf_model=hf_model,
                device=device,
                dtype=dtype,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
                **from_pretrained_kwargs,
            )
        else:
            raise

    model.eval()
    # Freeze params for interpretability work (no accidental grad)
    for param in model.parameters():
        param.requires_grad = False

    print(f"[model_loading] Loaded. n_layers={model.cfg.n_layers}, "
          f"d_model={model.cfg.d_model}, n_heads={model.cfg.n_heads}")
    return model


def get_residual_stream_layers(model: HookedTransformer) -> List[str]:
    """Return hook names for residual stream at each layer (post-layer-norm by default)."""
    # In transformer_lens, the standard residual stream hook is:
    # f"blocks.{i}.hook_resid_post"  (after attention + MLP, before next LN in many cfgs)
    # "blocks.{i}.hook_resid_pre" is also useful.
    return [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]


def get_layer_range_hooks(model: HookedTransformer, start: int = 8, end: Optional[int] = None) -> List[str]:
    """Get a sensible range of residual stream hooks (inclusive start, exclusive end).

    Per EXPERIMENTS.md guidance: middle layers often richest for concepts (e.g. 8-20 for 8B).
    """
    n_layers = model.cfg.n_layers
    if end is None or end > n_layers:
        end = n_layers
    if start < 0:
        start = 0
    return [f"blocks.{i}.hook_resid_post" for i in range(start, end)]


def estimate_memory_for_activations(
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_layers_sampled: int,
    dtype_bytes: int = 2,  # fp16/bf16
) -> Dict[str, float]:
    """Rough estimate of activation tensor memory (GB) for planning SAE training."""
    bytes_per_tensor = batch_size * seq_len * d_model * dtype_bytes
    total_bytes = bytes_per_tensor * n_layers_sampled
    gb = total_bytes / (1024 ** 3)
    return {
        "per_layer_gb": bytes_per_tensor / (1024 ** 3),
        "total_gb_approx": gb,
        "recommendation": (
            "Keep batch*seq*n_layers small (< 4-8GB target) on first runs. "
            "Use context filtering or short prompts for SAE data collection."
        ),
    }


if __name__ == "__main__":
    # Quick smoke test (will fail until packages + auth are present)
    print("model_loading.py smoke test (no model download unless you edit)")
    print("Supported short names:", list(SUPPORTED_MODELS.keys()))
    print("Device resolver test:", get_device("auto"))
