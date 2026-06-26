#!/usr/bin/env python3
"""
Initial causal analysis for key ontological features from L12 and L16 SAEs.
Focus on hasExtension_abstract, instanceOf, superset_subClass.
Especially feature 5532 (shared in hasExt) and strong catalog features.

Uses activation patching: ablate (zero) the SAE feature activation in the residual stream
and measure change in model output on simple reasoning prompts.

Run with:
  KMP_DUPLICATE_LIB_OK=TRUE python experiments/first_set/causal_ontological_features.py
"""

import os
import sys
from pathlib import Path
import torch
from functools import partial
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.model_loading import load_model
from src.sae.analyze_features import compute_feature_activations

# Key features identified from prior analysis and catalogs
KEY_FEATURES = {
    12: {
        "hasExtension_abstract": [5532, 46355],  # 5532 shared, 46355 high in catalog for mammal int/ext
        "instanceOf": [16736, 9555],  # Fido high in catalog
        "superset_subClass": [38995, 48351],  # even numbers subset high, mammals
    },
    16: {
        "hasExtension_abstract": [5532, 27422, 9049],  # 5532 shared, 27422 highest dog extension
        "instanceOf": [48684, 9882],  # Eiffel, whale instance high
        "superset_subClass": [48351, 17714],  # mammals superset, vehicles
    }
}

CANONICAL_TEST_PROMPTS = {
    "hasExtension": [
        "An abstract idea like 'justice' is different from the set of all just actions. The extension of justice includes",
        "The concept of 'dog' has an extension that includes all actual dogs. This means a poodle",
    ],
    "instanceOf": [
        "Fido is a specific instance of the class dog. Therefore Fido is a",
        "The Eiffel Tower is a specific instance of the abstract category 'landmark'. The Eiffel Tower is a",
    ],
    "superset_subClass": [
        "Mammals are a superset of dogs and cats. A poodle is a dog, therefore a poodle is a",
        "All birds are animals. A robin is a bird, therefore a robin is an",
    ],
}

def load_sae_for_layer(layer: int, device="cpu"):
    """Load the saved SAE for the layer, matching analyze_consistency logic."""
    base = Path("experiments/first_set/sae_runs/llama_3_1_8b_layer")
    dir_path = base.parent / f"llama_3_1_8b_layer{layer}_max"
    sae_path = dir_path / "sae.pt"
    if not sae_path.exists():
        dir_path = base.parent / f"llama_3_1_8b_layer{layer}"
        sae_path = dir_path / "sae.pt"
    print(f"Loading SAE for layer {layer} from {sae_path}")
    data = torch.load(sae_path, map_location=device)
    d_model = 4096
    d_sae = 49152
    try:
        from sae_lens import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
        cfg = BatchTopKTrainingSAEConfig(d_in=d_model, d_sae=d_sae, device=device, k=max(16, d_sae // 128))
        sae = BatchTopKTrainingSAE(cfg).to(device)
        sae.load_state_dict(data["state_dict"])
        print("Loaded as BatchTopKTrainingSAE")
        return sae
    except Exception as e:
        print(f"BatchTopK load failed ({e}), using simple wrapper")
        class SimpleSAE(torch.nn.Module):
            def __init__(self, d_in, d_sae):
                super().__init__()
                self.W_enc = torch.nn.Linear(d_in, d_sae, bias=True)
                self.W_dec = torch.nn.Linear(d_sae, d_in, bias=True)
                self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
            def encode(self, x):
                return torch.relu(self.W_enc(x) + self.b_enc)
            def decode(self, acts):
                return self.W_dec(acts)
            def forward(self, x):
                acts = self.encode(x)
                recon = self.decode(acts)
                return recon, acts
        sae = SimpleSAE(d_model, d_sae).to(device)
        sae.load_state_dict(data["state_dict"])
        return sae

def get_feature_acts(sae, resid):
    if hasattr(sae, 'encode'):
        return sae.encode(resid)
    # fallback
    return torch.relu(sae.W_enc(resid) + sae.b_enc)

def make_sae_feature_ablation_hook(sae, feat_idx, hook_name):
    """Return a hook fn that ablates (zeros) one SAE feature in the resid stream."""
    def hook_fn(resid_post, hook):
        # resid_post: [batch, pos, d_model]
        orig_shape = resid_post.shape
        flat = resid_post.reshape(-1, orig_shape[-1])
        feats = get_feature_acts(sae, flat)
        feats[:, feat_idx] = 0.0
        patched = sae.decode(feats) if hasattr(sae, 'decode') else sae.W_dec(feats)
        return patched.reshape(orig_shape)
    return hook_fn

def measure_logit_diff(model, prompt, correct_token, incorrect_token, hook=None, hook_name=None):
    """Run model (optionally with hook) and return logit(correct) - logit(incorrect) for last token."""
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        if hook is None:
            logits = model(tokens, return_type="logits")
        else:
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, hook)],
                return_type="logits"
            )
    last_logits = logits[0, -1]
    correct_id = model.to_single_token(correct_token)
    incorrect_id = model.to_single_token(incorrect_token)
    diff = last_logits[correct_id].item() - last_logits[incorrect_id].item()
    return diff

def main():
    print("=== Causal Analysis on Key Ontological Features (L12 + L16) ===")
    device = "cpu"
    model = load_model("llama-3.1-8b", device=device, dtype=torch.float32)
    print("Model loaded.")

    layers = [12, 16]
    results = {}

    for layer in layers:
        print(f"\n=== Layer {layer} ===")
        hook_name = f"blocks.{layer}.hook_resid_post"
        sae = load_sae_for_layer(layer, device=device)
        results[layer] = {}

        # Test features for the layer
        test_feats = []
        for cat, feats in KEY_FEATURES[layer].items():
            test_feats.extend(feats)
        test_feats = list(set(test_feats))  # unique

        for feat in test_feats[:5]:  # limit for speed on CPU
            print(f"Testing feature {feat} ...")
            ablation_hook = make_sae_feature_ablation_hook(sae, feat, hook_name)

            layer_res = {}
            for cat_name, prompts in CANONICAL_TEST_PROMPTS.items():
                clean_diffs = []
                patched_diffs = []
                for p in prompts:
                    # Use simple correct/incorrect for the relation
                    if "poodle" in p.lower() and "mammal" in cat_name.lower() or "superset" in cat_name.lower():
                        correct = " mammal"
                        incorrect = " cat"
                    elif "Fido" in p or "instance" in cat_name.lower():
                        correct = " dog"
                        incorrect = " cat"
                    else:
                        correct = " dogs"
                        incorrect = " cats"

                    try:
                        clean_diff = measure_logit_diff(model, p, correct, incorrect)
                        patched_diff = measure_logit_diff(model, p, correct, incorrect, ablation_hook, hook_name)
                        clean_diffs.append(clean_diff)
                        patched_diffs.append(patched_diff)
                    except Exception as e:
                        print(f"  Skipped prompt due to token issue: {e}")
                        continue

                if clean_diffs:
                    delta = [p - c for p, c in zip(patched_diffs, clean_diffs)]
                    avg_delta = sum(delta) / len(delta)
                    layer_res[cat_name] = {
                        "avg_clean_diff": sum(clean_diffs)/len(clean_diffs),
                        "avg_patched_diff": sum(patched_diffs)/len(patched_diffs),
                        "avg_effect_of_ablation": avg_delta
                    }
                    print(f"  {cat_name}: clean_diff={layer_res[cat_name]['avg_clean_diff']:.2f}, "
                          f"patched_diff={layer_res[cat_name]['avg_patched_diff']:.2f}, "
                          f"ablation effect={avg_delta:.2f}")

            results[layer][feat] = layer_res

    print("\n=== Summary ===")
    for lyr, feats in results.items():
        print(f"Layer {lyr}:")
        for f, cats in feats.items():
            for cat, vals in cats.items():
                effect = vals.get('avg_effect_of_ablation', 0)
                direction = "decreases correct preference" if effect < 0 else "increases or no change"
                print(f"  Feature {f} on {cat}: ablation effect {effect:.2f} ({direction})")

    # Save results
    out_path = Path("experiments/first_set/sae_runs/causal_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {out_path}")

if __name__ == "__main__":
    main()
