#!/usr/bin/env python3
"""
Maximum aggression collection + SAE training for Phase 1.

Collects very large numbers of activations (200k-300k+ tokens per layer) 
for layers 12,16,20 on Llama-3.1-8B.

Uses efficient batching, mixed data, and trains large BatchTopK SAEs.

Run with:
  python experiments/first_set/max_aggression_collect.py --tokens 250000 --layers 12,16,20

This is designed to be the hardest feasible push for feature discovery.
"""

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.model_loading import load_model
from src.sae.train_sae import train_sae_on_activations
from src.sae.analyze_features import (
    compute_feature_activations,
    find_top_examples_per_feature,
    summarize_features,
    save_feature_catalog,
)

# Expanded ontological + general prompts for max signal + diversity
PROMPTS = [
    # hasExtension / abstract vs extension
    "The concept of 'dog' has an extension that includes all actual dogs that exist or have existed.",
    "An abstract idea like 'justice' is different from the set of all just actions.",
    "The intension of 'mammal' is the abstract category; its extension is every living mammal.",
    # superset / subClass
    "A poodle is a dog. A dog is a mammal. Therefore poodles are mammals.",
    "All birds are animals, but not all animals are birds.",
    "Vehicles include cars, boats, and airplanes as subclasses.",
    # instanceOf / hasElement
    "Fido is a specific instance of the class dog.",
    "The Eiffel Tower is an instance of 'landmark'.",
    "Earth is an element of the set of planets.",
    # Mixed relational language
    "Mammals are a superset of dogs and cats.",
    "The set of even numbers is a proper subset of the integers.",
    "Professions such as doctor and lawyer are types of occupations.",
    "The word 'mammal' refers to the abstract category, while this particular whale is an instance.",
]

def collect_activations_aggressive(model, hooks, n_tokens=250000, batch_size=32, context=128):
    """Aggressive collection: large batch, mixed prompts + pile, progress reporting."""
    full_texts = []
    target = n_tokens * 4
    while len(full_texts) * context < target // 2:
        full_texts.extend(PROMPTS)

    print(f"Mixing real data from pile-10k...")
    try:
        ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
        for ex in ds.take(5000):  # more data for max aggression
            text = ex.get("text", "")[:600]
            if len(text) > 30:
                full_texts.append(text)
                if len(full_texts) * context > target:
                    break
    except Exception as e:
        print(f"  Pile mix limited: {e}")

    print(f"Collecting ~{n_tokens} tokens per hook across {len(hooks)} layers (batch={batch_size})...")

    collected = {h: [] for h in hooks}
    labels = {h: [] for h in hooks}

    for i in range(0, len(full_texts), batch_size):
        batch = full_texts[i : i + batch_size]
        if not batch:
            break
        with torch.no_grad():
            _, cache = model.run_with_cache(batch, names_filter=hooks, return_type=None)
        for hook in hooks:
            acts = cache[hook].reshape(-1, cache[hook].shape[-1]).cpu()
            collected[hook].append(acts)
            seq_len = acts.shape[0] // len(batch)
            for txt in batch:
                labels[hook].extend([txt[:300]] * seq_len)

        if i % (batch_size * 10) == 0:
            total = sum(len(x) for x in collected[hooks[0]])
            print(f"  progress: ~{total} tokens collected so far")

    results = {}
    for hook in hooks:
        acts = torch.cat(collected[hook], dim=0)[:n_tokens]
        lbl = labels[hook][:n_tokens]
        results[hook] = (acts, lbl)
        print(f"  {hook}: final {acts.shape[0]} vectors")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=250000)
    parser.add_argument("--layers", default="12,16,20")
    parser.add_argument("--expansion", type=int, default=16)
    args = parser.parse_args()

    layer_list = [int(x) for x in args.layers.split(",")]
    hooks = [f"blocks.{l}.hook_resid_post" for l in layer_list]

    print("MAX AGGRESSION PHASE 1 COLLECTOR")
    print(f"Layers: {layer_list} | Tokens/layer: {args.tokens} | Expansion: {args.expansion}x")

    model = load_model("llama-3.1-8b", device="cpu")  # cpu for stability on long run
    d_model = model.cfg.d_model

    data = collect_activations_aggressive(model, hooks, n_tokens=args.tokens)

    for hook in hooks:
        acts, lbls = data[hook]
        d_sae = int(d_model * args.expansion)
        print(f"\nTraining large BatchTopK SAE for {hook} (d_sae={d_sae})...")
        sae = train_sae_on_activations(
            acts, d_model, d_sae=d_sae, lr=1e-4, steps=1200, batch_size=256, device="cpu", seed=42
        )

        print("Analyzing...")
        fa = compute_feature_activations(sae, acts[: min(8192, len(acts))])
        tops = find_top_examples_per_feature(fa, lbls[: min(8192, len(lbls))], top_k=5)
        sums = summarize_features(fa, tops)

        safe = f"llama_3_1_8b_layer{hook.split('.')[1]}_max"
        out = Path(f"experiments/first_set/sae_runs/{safe}")
        out.mkdir(parents=True, exist_ok=True)
        save_feature_catalog(sums, out / "feature_catalog.json", {
            "model": "llama-3.1-8b", "hook": hook, "tokens": args.tokens, "expansion": args.expansion
        })
        torch.save({"state_dict": sae.state_dict()}, out / "sae.pt")
        print(f"Saved to {out}")

    print("\nMax aggression collection + training complete.")

if __name__ == "__main__":
    main()
