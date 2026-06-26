"""
Feature analysis utilities for Phase 1.

After training an SAE (or loading pretrained one), this module helps:
- Find top-activating examples for each feature (on a held-out set of texts/activations)
- Compute basic statistics (sparsity, max activation, etc.)
- Prepare data for automated interpretability (feature descriptions)

This is intentionally simple and focused. Later we can add:
- Max activating dataset examples with token highlighting
- Cosine similarity between features
- Cluster analysis of features (category vs instance signals)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Iterable, Union
from pathlib import Path
import json

try:
    import torch
except ImportError:
    torch = None  # type: ignore

import numpy as np

try:
    from transformer_lens import HookedTransformer
except ImportError:
    HookedTransformer = None  # type: ignore


@dataclass
class FeatureExample:
    feature_idx: int
    activation: float
    text: str
    # token-level info can be added later
    tokens: Optional[List[str]] = None
    token_acts: Optional[List[float]] = None


@dataclass
class FeatureSummary:
    feature_idx: int
    max_activation: float
    mean_activation: float
    sparsity: float  # fraction of examples with >0 activation (or > threshold)
    top_examples: List[FeatureExample]
    description: Optional[str] = None  # filled by auto-interp or manual


def compute_feature_activations(
    sae: Any,
    activations: "torch.Tensor",
    batch_size: int = 256,
) -> "torch.Tensor":
    """
    Run the SAE encoder on a large batch of activations [N, d_model] -> [N, d_sae]
    Returns feature activation matrix (post-ReLU or equivalent).
    """
    if torch is None:
        raise ImportError("torch required")
    device = next(sae.parameters()).device if hasattr(sae, "parameters") else "cpu"
    acts = activations.to(device)
    all_feature_acts = []

    with torch.no_grad():
        for i in range(0, len(acts), batch_size):
            batch = acts[i : i + batch_size]
            # Different SAE implementations expose different forward signatures.
            # Try common ones:
            if hasattr(sae, "encode"):
                feature_acts = sae.encode(batch)
            elif hasattr(sae, "forward"):
                out = sae(batch)
                # Common returns: (recon, feature_acts) or dict
                if isinstance(out, tuple) and len(out) >= 2:
                    feature_acts = out[1]
                elif isinstance(out, dict):
                    feature_acts = out.get("feature_acts", out.get("acts", out.get("sparse", None)))
                else:
                    feature_acts = out
            else:
                # Assume it's a tiny fallback SAE from train_sae
                feature_acts = torch.relu(sae.W_enc(batch) + sae.b_enc) if hasattr(sae, "W_enc") else batch @ sae.W_enc.weight.T
            all_feature_acts.append(feature_acts.cpu())

    return torch.cat(all_feature_acts, dim=0)


def find_top_examples_per_feature(
    feature_activations: torch.Tensor,
    texts: List[str],
    top_k: int = 5,
    min_activation: float = 0.5,
) -> Dict[int, List[FeatureExample]]:
    """
    For each feature, return the top_k examples (by max activation on that feature).

    feature_activations: [N, d_sae]
    texts: list of length N (corresponding prompts or sentences)
    """
    if torch is None:
        raise ImportError("torch required for find_top_examples_per_feature")

    n_features = feature_activations.shape[1]
    results: Dict[int, List[FeatureExample]] = {}

    for f in range(n_features):
        acts_f = feature_activations[:, f]
        # filter to reasonably activating
        mask = acts_f > min_activation
        if mask.sum() == 0:
            continue
        vals = acts_f[mask]
        idxs = torch.arange(len(acts_f))[mask]

        # top k within this feature
        top_vals, top_local = torch.topk(vals, k=min(top_k, len(vals)))
        top_idxs = idxs[top_local]

        examples = []
        for val, gidx in zip(top_vals.tolist(), top_idxs.tolist()):
            examples.append(
                FeatureExample(
                    feature_idx=f,
                    activation=float(val),
                    text=texts[gidx],
                )
            )
        results[f] = examples

    return results


def summarize_features(
    feature_activations: torch.Tensor,
    top_examples: Dict[int, List[FeatureExample]],
    activation_threshold: float = 0.1,
) -> List[FeatureSummary]:
    """Aggregate basic stats per feature."""
    summaries = []
    n, d_sae = feature_activations.shape

    for f in range(d_sae):
        acts_f = feature_activations[:, f]
        max_a = float(acts_f.max().item())
        mean_a = float(acts_f.mean().item())
        non_zero = (acts_f > activation_threshold).float().mean().item()

        exs = top_examples.get(f, [])
        summaries.append(
            FeatureSummary(
                feature_idx=f,
                max_activation=max_a,
                mean_activation=mean_a,
                sparsity=float(non_zero),
                top_examples=exs,
            )
        )
    return summaries


def save_feature_catalog(
    summaries: List[FeatureSummary],
    path: Union[str, Path],
    model_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Save human-readable + machine readable catalog of features."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = []
    for s in summaries:
        d = {
            "feature_idx": s.feature_idx,
            "max_activation": s.max_activation,
            "mean_activation": s.mean_activation,
            "sparsity": s.sparsity,
            "top_examples": [
                {"activation": e.activation, "text": e.text} for e in s.top_examples
            ],
            "description": s.description,
        }
        serializable.append(d)

    payload = {
        "model_info": model_info or {},
        "num_features_analyzed": len(summaries),
        "features": serializable,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    # Also write a short human summary
    md_path = path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("# SAE Feature Catalog (Phase 1)\n\n")
        if model_info:
            f.write(f"Model: {model_info}\n\n")
        f.write(f"Total features scanned: {len(summaries)}\n\n")
        # Show a few highest max-activation as example
        top_by_max = sorted(summaries, key=lambda x: x.max_activation, reverse=True)[:10]
        f.write("## Highest max-activation features (sample)\n\n")
        for s in top_by_max:
            f.write(f"**Feature {s.feature_idx}** — max={s.max_activation:.2f}, sparsity={s.sparsity:.3f}\n")
            for ex in s.top_examples[:2]:
                short = ex.text[:120].replace("\n", " ")
                f.write(f"  - act={ex.activation:.2f}: {short}...\n")
            f.write("\n")
    print(f"[analyze] Saved catalog to {path} and {md_path}")


def filter_for_ontological_interest(
    summaries: List[FeatureSummary],
    min_max_act: float = 2.0,
    max_sparsity: float = 0.15,
) -> List[FeatureSummary]:
    """
    Very rough heuristic filter for features that might be interesting
    for categories / instances / relations.

    Real work will use better auto-interp + manual review.
    """
    interesting = []
    for s in summaries:
        if s.max_activation >= min_max_act and s.sparsity <= max_sparsity:
            interesting.append(s)
    return interesting


if __name__ == "__main__":
    print("analyze_features.py ready. Use from notebook or scripts.")
    print("Example: load cached activations + trained SAE, then compute summaries.")
