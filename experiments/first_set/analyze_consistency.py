#!/usr/bin/env python3
"""
Post-training consistency / LTC-style analysis for Phase 1.

After running aggressive SAE training on multiple layers, this script:
- Loads the saved SAEs (sae.pt) + catalogs from a run.
- Re-collects or uses activations for canonical ontological test cases.
- For each relation family, finds the strongest features per layer.
- Computes overlap (top-k Jaccard) and reports shared/reusable features.
- This is an early quantitative step toward the LLM-Tapestry Characteristic (LTC).

Usage after a run:
    python experiments/first_set/analyze_consistency.py \
        --run_base experiments/first_set/sae_runs/llama_3_1_8b_layer \
        --layers 12 16 20

The script auto-detects layer dirs with or without "_max" suffix.
For a single completed layer (e.g. 16):
    python experiments/first_set/analyze_consistency.py \
        --run_base experiments/first_set/sae_runs/llama_3_1_8b_layer \
        --layers 16
"""

import argparse
import sys
from pathlib import Path
import torch
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.model_loading import load_model
from src.sae.analyze_features import compute_feature_activations

CANONICAL_CASES = {
    "hasExtension_abstract": [
        "The concept of 'dog' has an extension that includes all actual dogs.",
        "An abstract idea like 'justice' is different from the set of all just actions and institutions.",
    ],
    "instanceOf": [
        "Fido is an instance of dog.",
        "The Eiffel Tower is a specific instance of the abstract category 'landmark'.",
        "Earth is an instance of planet.",
    ],
    "superset_subClass": [
        "All birds are animals, but not all animals are birds.",
        "Mammals are a superset of dogs and cats.",
        "A poodle is a dog, and a dog is a mammal.",
    ],
    "category_vs_member": [
        "The word 'mammal' refers to the abstract category, while 'this particular whale' refers to an instance.",
        "The concept 'vehicle' versus this specific red bicycle.",
    ],
}

def load_sae(sae_path: Path, d_in: int, d_sae: int, device: str = "cpu"):
    """Load a saved SAE from our simple torch.save format."""
    data = torch.load(sae_path, map_location=device)
    # Try to use BatchTopK if available, else fall back to a simple wrapper
    try:
        from sae_lens import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
        cfg = BatchTopKTrainingSAEConfig(d_in=d_in, d_sae=d_sae, device=device, k=max(16, d_sae // 128))
        sae = BatchTopKTrainingSAE(cfg).to(device)
        sae.load_state_dict(data["state_dict"])
        return sae
    except Exception:
        # Minimal fallback decoder/encoder
        class SimpleSAE(torch.nn.Module):
            def __init__(self, d_in, d_sae):
                super().__init__()
                self.W_enc = torch.nn.Linear(d_in, d_sae, bias=True)
                self.W_dec = torch.nn.Linear(d_sae, d_in, bias=True)
                self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
            def forward(self, x):
                acts = torch.relu(self.W_enc(x) + self.b_enc)
                recon = self.W_dec(acts)
                return recon, acts
        sae = SimpleSAE(d_in, d_sae).to(device)
        sae.load_state_dict(data["state_dict"])
        return sae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_base", type=str, default="experiments/first_set/sae_runs/llama_3_1_8b_layer",
                        help="Base path prefix for per-layer dirs, e.g. ..._layer")
    parser.add_argument("--layers", nargs="+", type=int, default=[12,16,20])
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    print("=== Phase 1 LTC-style Consistency Analysis ===")
    model = load_model("llama-3.1-8b", device="cpu")
    d_model = model.cfg.d_model

    def _get_layer_dir(base: str, lyr: int) -> Path:
        """Support both 'layer{lyr}' and 'layer{lyr}_max' (and minor variants) used by collectors."""
        cands = [
            Path(f"{base}{lyr}_max"),
            Path(f"{base}{lyr}"),
            Path(f"{base}_{lyr}"),
            Path(str(base).rstrip("_") + f"_{lyr}"),
        ]
        for c in cands:
            if (c / "sae.pt").exists():
                return c
        return Path(f"{base}{lyr}_max")

    layer_data = {}
    for lyr in args.layers:
        hook = f"blocks.{lyr}.hook_resid_post"
        dir_path = _get_layer_dir(args.run_base, lyr)
        cat_path = dir_path / "feature_catalog.json"
        sae_path = dir_path / "sae.pt"

        if not sae_path.exists():
            print(f"Warning: no sae.pt in {dir_path}, skipping layer {lyr}")
            continue

        # Load catalog for d_sae
        if cat_path.exists():
            with open(cat_path) as f:
                meta = json.load(f).get("model_info", {})
            d_sae = meta.get("d_sae") or int(4096 * meta.get("expansion", 8))
        else:
            d_sae = 32768

        print(f"Loading SAE for layer {lyr} (d_sae={d_sae}) ...")
        sae = load_sae(sae_path, d_model, d_sae, device="cpu")

        # Collect fresh activations for the canonical cases
        test_texts = []
        for grp, prompts in CANONICAL_CASES.items():
            test_texts.extend(prompts)

        with torch.no_grad():
            _, cache = model.run_with_cache(test_texts, names_filter=[hook], return_type=None)
        acts = cache[hook].reshape(-1, d_model).cpu()

        feat_acts = compute_feature_activations(sae, acts, batch_size=64)
        # For each group, find strongest features
        group_top = {}
        start = 0
        for grp, prompts in CANONICAL_CASES.items():
            n = len(prompts) * acts.shape[0] // len(test_texts)   # rough per-group slice
            grp_acts = feat_acts[start:start+n]
            max_per_feat = grp_acts.max(dim=0).values
            top_idx = torch.topk(max_per_feat, k=min(args.top_k, max_per_feat.numel())).indices.tolist()
            group_top[grp] = set(top_idx)
            start += n

        layer_data[lyr] = {"sae": sae, "top_per_group": group_top, "hook": hook}

    # Per-layer top features (useful even for single-layer runs)
    print("\n=== Top feature ids per ontological group (by max activation on canonical prompts) ===")
    for lyr in sorted(layer_data):
        print(f"\nLayer {lyr}:")
        for grp, idxs in layer_data[lyr]["top_per_group"].items():
            top_list = sorted(list(idxs))[:15]
            print(f"  {grp}: {top_list}{' ...' if len(idxs) > 15 else ''}")

    # Cross-layer overlap per group
    print("\n=== Cross-layer feature overlap (Jaccard @ top-k) ===")
    for grp in CANONICAL_CASES:
        sets = [layer_data[l]["top_per_group"][grp] for l in sorted(layer_data) if grp in layer_data[l]["top_per_group"]]
        if len(sets) >= 2:
            print(f"\n{grp}:")
            layers = sorted([l for l in layer_data if grp in layer_data[l]["top_per_group"]])
            for i in range(len(layers)):
                for j in range(i+1, len(layers)):
                    a, b = sets[i], sets[j]
                    inter = len(a & b)
                    union = len(a | b)
                    jac = inter / union if union > 0 else 0
                    print(f"  layer {layers[i]} vs {layers[j]}: Jaccard={jac:.3f} (overlap {inter}/{union})")

    print("\nAnalysis complete. Look for groups with persistently high overlap across layers — candidate 'reusable' ontological features.")

if __name__ == "__main__":
    main()
