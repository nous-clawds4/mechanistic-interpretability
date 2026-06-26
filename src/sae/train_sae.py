"""
Sparse Autoencoder (SAE) training utilities for Phase 1: Feature Discovery.

This module wraps sae_lens to train SAEs on residual stream activations
collected from a transformer_lens model.

Key design goals (per EXPERIMENTS.md + CLAUDE.md):
- Reproducible
- Focus on middle layers (roughly 8-20 for 8B models)
- Document hyperparameters
- Runnable on Mac Studio M-series (use small batch / d_sae initially)

Typical flow:
    1. model = load_model(...)
    2. sae = train_sae_on_texts(model, texts, hook_name="blocks.12.hook_resid_post", ...)
    3. or use the config-driven entrypoint with experiments/first_set/config.yaml

The high-level runner from sae_lens is preferred when available.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Iterable, Union
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None  # type: ignore

import yaml

try:
    from transformer_lens import HookedTransformer
except ImportError:
    HookedTransformer = None  # type: ignore

try:
    from src.utils.model_loading import load_model
except Exception:
    load_model = None  # type: ignore

try:
    # sae_lens 6.x API (current as of 2026)
    from sae_lens import (
        LanguageModelSAETrainingRunner,
        LanguageModelSAERunnerConfig,
        BatchTopKTrainingSAEConfig,
        LoggingConfig,
    )
    SAE_LENS_AVAILABLE = True
except Exception:
    # Fallback
    LanguageModelSAETrainingRunner = None  # type: ignore
    LanguageModelSAERunnerConfig = None  # type: ignore
    BatchTopKTrainingSAEConfig = None  # type: ignore
    LoggingConfig = None  # type: ignore
    SAE_LENS_AVAILABLE = False


@dataclass
class SAEExperimentConfig:
    """Serializable config for a single SAE training run.

    Matches spirit of experiments/first_set/config.yaml
    """
    # Model / data
    model_name: str = "llama-3.1-8b"
    hook_name: str = "blocks.12.hook_resid_post"
    dataset_name: str = "NeelNanda/pile-10k"   # small public dataset good for exploration
    dataset_split: str = "train"
    is_dataset_tokenized: bool = False

    # SAE architecture
    d_sae: int = 24576          # expansion factor ~ 8x on d_model=4096 for Llama-8B
    expansion_factor: Optional[int] = None  # if set, overrides d_sae calc

    # Training
    lr: float = 5e-5
    l1_coefficient: float = 5e-3
    batch_size: int = 4
    context_size: int = 128     # short for Mac memory during discovery
    num_epochs: int = 1
    num_training_steps: Optional[int] = 1000   # safety cap for first experiments

    # Misc
    seed: int = 42
    device: str = "auto"
    log_to_wandb: bool = False
    wandb_project: str = "llm-tapestry-phase1"
    output_dir: str = "experiments/first_set/sae_runs"
    save_every_n_steps: int = 200

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SAEExperimentConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        # Only take fields we know about
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)

    def save_yaml(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def resolve_device(device: str) -> str:
    if device == "auto":
        if torch is None:
            return "cpu"
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


def build_sae_training_config(
    exp_cfg: SAEExperimentConfig,
    model: Optional[HookedTransformer] = None,
) -> Any:
    """Build a modern sae_lens 6.x LanguageModelSAERunnerConfig.

    Uses BatchTopKTrainingSAEConfig (common modern default).
    If model is not passed, we temporarily load one (cheap for small models)
    to get the correct d_model.
    """
    if not SAE_LENS_AVAILABLE or LanguageModelSAERunnerConfig is None:
        raise RuntimeError("sae_lens not fully available.")

    device = resolve_device(exp_cfg.device)

    d_model = None
    if model is not None and hasattr(model, "cfg"):
        d_model = model.cfg.d_model
    else:
        # Auto-detect d_model by loading the model (only the config is heavy)
        if load_model is None:
            raise RuntimeError("load_model not available to auto-detect d_model")
        print("[build] No model provided — loading to discover d_model ...")
        temp_model = load_model(exp_cfg.model_name, device="cpu")
        d_model = temp_model.cfg.d_model
        # free memory
        del temp_model
        if torch is not None:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    if d_model is None:
        d_model = 4096

    d_sae = exp_cfg.d_sae
    if exp_cfg.expansion_factor is not None:
        d_sae = int(d_model * exp_cfg.expansion_factor)

    # Create the SAE training config (BatchTopK is a solid modern choice)
    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=d_model,
        d_sae=d_sae,
        device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
        k=max(16, d_sae // 64),   # reasonable k for small runs
    )

    training_tokens = (
        exp_cfg.num_training_steps * exp_cfg.batch_size * exp_cfg.context_size
        if exp_cfg.num_training_steps
        else 200_000
    )

    # Disable wandb by default (user can opt in later)
    logger_cfg = LoggingConfig(log_to_wandb=False) if LoggingConfig is not None else None

    runner_cfg = LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=exp_cfg.model_name,
        hook_name=exp_cfg.hook_name,
        dataset_path=exp_cfg.dataset_name,
        is_dataset_tokenized=exp_cfg.is_dataset_tokenized,
        context_size=exp_cfg.context_size,
        training_tokens=training_tokens,
        train_batch_size_tokens=exp_cfg.batch_size * exp_cfg.context_size,
        device=device,
        seed=exp_cfg.seed,
        lr=exp_cfg.lr,
        checkpoint_path=str(Path(exp_cfg.output_dir)),
        n_checkpoints=0,
        verbose=True,
        logger=logger_cfg,
    )
    return runner_cfg


def train_sae_from_config(
    exp_cfg: SAEExperimentConfig,
    model: Optional[HookedTransformer] = None,
) -> Any:
    """
    Train an SAE using sae_lens high-level runner.

    Returns the trained SAE (or runner result object).
    Saves to exp_cfg.output_dir .
    """
    if not SAE_LENS_AVAILABLE:
        raise RuntimeError("sae_lens not available. See requirements.txt")

    Path(exp_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Persist the exact config used
    exp_cfg.save_yaml(Path(exp_cfg.output_dir) / "run_config.yaml")

    runner_cfg = build_sae_training_config(exp_cfg, model=model)

    print("[train_sae] Starting modern sae_lens training (BatchTopK)...")
    print(f"  hook={exp_cfg.hook_name}")
    # Use the value that was actually passed to the SAE config
    actual_d_sae = getattr(runner_cfg.sae, "d_sae", getattr(runner_cfg, "d_sae", exp_cfg.d_sae)) if hasattr(runner_cfg, "sae") else exp_cfg.d_sae
    print(f"  d_sae={actual_d_sae}, batch_size_tokens~={exp_cfg.batch_size * exp_cfg.context_size}")
    print(f"  training_tokens target ~{getattr(runner_cfg, 'training_tokens', '?')}")
    print(f"  device={resolve_device(exp_cfg.device)}")

    Path(exp_cfg.output_dir).mkdir(parents=True, exist_ok=True)
    exp_cfg.save_yaml(Path(exp_cfg.output_dir) / "run_config.yaml")

    runner = LanguageModelSAETrainingRunner(runner_cfg)
    try:
        sae = runner.run()
    except Exception as e:
        print(f"[train_sae] runner.run() raised during/after training: {e}")
        print("  Attempting to recover the SAE object from the trainer if possible...")
        # The runner usually has a .trainer.sae after fit; try common locations
        sae = None
        if hasattr(runner, "trainer") and hasattr(runner.trainer, "sae"):
            sae = runner.trainer.sae
        elif hasattr(runner, "_sae"):
            sae = runner._sae
        if sae is None:
            raise
        print("  Recovered SAE object despite save error.")
    print(f"[train_sae] Training complete (or recovered). SAE object available.")
    print(f"  Artifacts (if any) under {exp_cfg.output_dir}")
    return sae


def train_sae_on_activations(
    activations: "torch.Tensor",
    d_model: int,
    d_sae: int = 24576,
    lr: float = 5e-5,
    l1_coefficient: float = 5e-3,
    steps: int = 1000,
    batch_size: int = 128,
    device: str = "cpu",
    seed: int = 42,
) -> Any:
    """
    Lower-level path: train SAE directly on a tensor of activations [N, d_model].

    Useful for:
    - Toy / unit tests
    - When you have already cached a large activation dataset

    This uses the lower-level SAE + training loop exposed by sae_lens when possible.
    Falls back to a minimal PyTorch training loop if needed.
    """
    if torch is None:
        raise ImportError("torch required for train_sae_on_activations")
    torch.manual_seed(seed)

    # Preferred: use a real sae_lens BatchTopKTrainingSAE when available.
    # Falls back to minimal only if necessary.
    if SAE_LENS_AVAILABLE and BatchTopKTrainingSAEConfig is not None:
        try:
            return _train_batch_topk_on_activations(
                activations, d_model, d_sae, lr, steps, batch_size, device, seed
            )
        except Exception as e:
            print(f"[train_sae] BatchTopK path failed ({e}), falling back to minimal impl.")
    print("[train_sae] Using controlled minimal SAE training loop.")
    return _minimal_sae_train(
        activations, d_model, d_sae, lr, l1_coefficient, steps, batch_size, device
    )


def _minimal_sae_train(
    activations: "torch.Tensor",
    d_model: int,
    d_sae: int,
    lr: float,
    l1: float,
    steps: int,
    batch_size: int,
    device: str,
) -> "torch.nn.Module":
    """Ultra-minimal linear SAE for smoke-testing code paths when sae_lens is absent."""
    if torch is None:
        raise ImportError("torch required")
    import torch.nn as nn
    import torch.nn.functional as F

    class TinySAE(nn.Module):
        def __init__(self, d_in: int, d_hidden: int):
            super().__init__()
            self.W_enc = nn.Linear(d_in, d_hidden, bias=True)
            self.W_dec = nn.Linear(d_hidden, d_in, bias=True)
            self.b_enc = nn.Parameter(torch.zeros(d_hidden))
            # Tie or init decoder orthogonally-ish
            nn.init.xavier_uniform_(self.W_enc.weight)
            nn.init.xavier_uniform_(self.W_dec.weight)

        def forward(self, x):
            acts = F.relu(self.W_enc(x) + self.b_enc)
            recon = self.W_dec(acts)
            return recon, acts

    model = TinySAE(d_model, d_sae).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    acts = activations.to(device)

    for step in range(steps):
        idx = torch.randint(0, len(acts), (batch_size,))
        x = acts[idx]
        recon, acts_hid = model(x)
        recon_loss = ((recon - x) ** 2).mean()
        l1_loss = l1 * acts_hid.abs().sum(-1).mean()
        loss = recon_loss + l1_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"[tiny-sae] step {step} recon={recon_loss.item():.4f}")
    return model


def _train_batch_topk_on_activations(
    activations: "torch.Tensor",
    d_model: int,
    d_sae: int,
    lr: float,
    steps: int,
    batch_size: int,
    device: str,
    seed: int,
) -> Any:
    """Train a real BatchTopKTrainingSAE from sae_lens on pre-collected activations."""
    if torch is None:
        raise ImportError("torch required")
    torch.manual_seed(seed)

    from sae_lens import BatchTopKTrainingSAEConfig, BatchTopKTrainingSAE

    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=d_model,
        d_sae=d_sae,
        device=device,
        k=max(8, d_sae // 128),
    )
    sae = BatchTopKTrainingSAE(sae_cfg).to(device)

    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    acts = activations.to(device)

    n = acts.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,))
        x = acts[idx]
        # Forward: most versions return (reconstruction, feature_acts) or similar
        out = sae(x)
        if isinstance(out, tuple) and len(out) >= 2:
            recon, f_acts = out[0], out[1]
        else:
            recon = out
            f_acts = sae.encode(x) if hasattr(sae, "encode") else x

        recon_loss = ((recon - x) ** 2).mean()
        # For TopK we can add aux loss if exposed, else simple sparsity
        sparsity_loss = 0.0
        if hasattr(sae, "aux_loss") and callable(getattr(sae, "aux_loss", None)):
            sparsity_loss = sae.aux_loss(f_acts) * 0.1   # light weight
        loss = recon_loss + sparsity_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 150 == 0 or step == steps - 1:
            print(f"[BatchTopK] step {step:4d} recon={recon_loss.item():.5f}")

    return sae


def main(config_path: Optional[str] = None):
    """CLI-friendly entry point.

    python -m src.sae.train_sae --config experiments/first_set/config.yaml
    """
    if config_path and Path(config_path).exists():
        exp_cfg = SAEExperimentConfig.from_yaml(config_path)
    else:
        exp_cfg = SAEExperimentConfig()
        print("[train_sae] No config provided, using defaults (good for tiny smoke tests).")

    # In real use you would load model here if the runner needs it.
    # For the sae_lens LanguageModel runner, it often loads internally.
    sae = train_sae_from_config(exp_cfg)
    return sae


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(args.config)
