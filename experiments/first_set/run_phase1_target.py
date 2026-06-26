#!/usr/bin/env python3
"""
Phase 1 Feature Discovery — AGGRESSIVE run for primary target models.

Primary targets (per EXPERIMENTS.md):
- Llama-3.1-8B-Instruct
- Mistral-7B-Instruct-v0.3

This script is now configured for maximum aggressiveness within reason:
- Multiple mid-to-late layers (12, 16, 20)
- Large token budgets (100k–200k+ per layer)
- Rich, diverse prompt set covering many domains and relation phrasings
- Proper BatchTopK SAEs with good expansion
- Basic consistency analysis across layers/concepts

Usage (aggressive example):
    python experiments/first_set/run_phase1_target.py --model llama-3.1-8b --layers 12,16,20 --tokens 150000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.utils.model_loading import load_model, get_layer_range_hooks
from src.sae.train_sae import train_sae_on_activations
from src.sae.analyze_features import (
    compute_feature_activations,
    find_top_examples_per_feature,
    summarize_features,
    save_feature_catalog,
)

TARGET_LAYERS_DEFAULT = [12, 16, 20]   # aggressive: mid-to-late layers for conceptual structure
CONTEXT = 128
TRAINING_TOKENS_TARGET = 200_000   # maximum aggression: 200k tokens per layer
BATCH_SIZE = 16                    # push collection speed (M3 Ultra has headroom)
D_SAE_EXPANSION = 16               # very large SAEs for max feature resolution (65k features)

# Massive diverse prompt set for intension/extension, hierarchy, membership across domains.
# Goal: strong signal for hasExtension, supersetOf/subClassOf, hasElement/instanceOf.
PROMPTS = [
    # === Biology / Animals ===
    "The concept of 'mammal' has an extension consisting of every actual mammal.",
    "A poodle is a dog. Every dog is a mammal. Therefore a poodle is a mammal.",
    "Fido is a specific golden retriever. He is an instance of the category dog.",
    "Whales, dolphins, and bats are all instances of the abstract class mammal.",
    "The set of all dogs is a proper subset of the set of all mammals.",
    "Not every animal is a mammal; birds and reptiles are animals that are not mammals.",

    # === Vehicles / Artifacts ===
    "The word 'vehicle' refers to the abstract category whose extension includes cars, trucks, boats, and planes.",
    "A Tesla Model 3 is a concrete instance of the class car.",
    "All cars are vehicles, but not every vehicle is a car.",
    "Bicycles and skateboards are also members of the broader set of vehicles.",

    # === Professions / Social categories ===
    "Doctor, nurse, and surgeon are all instances of the profession 'medical practitioner'.",
    "The abstract concept 'teacher' has thousands of individual instances in any large city.",
    "Every lawyer is a professional, but not every professional is a lawyer.",
    "Professions form a hierarchy: surgeon is a subtype of doctor.",

    # === Abstract concepts vs their extensions ===
    "The idea of 'justice' is an intension; the collection of all just actions and institutions is its extension.",
    "Democracy as an abstract political system versus the set of all actual democratic countries.",
    "The number 'seven' is an abstract mathematical object whose extension in the physical world is hard to define.",
    "Beauty is an abstract quality; beautiful objects form its extensional set.",

    # === Sets, math, logic ===
    "The set of all even positive integers is a proper subset of the natural numbers.",
    "Prime numbers are a subset of the integers. 2, 3, 5, and 7 are specific instances.",
    "A square is a special case of a rectangle. Every square is a rectangle.",
    "The extension of the predicate 'is a power of two' includes 1, 2, 4, 8, 16, ...",

    # === Food / Natural kinds ===
    "Apple, banana, and orange are instances of the category fruit.",
    "All fruits are plant products, but not all plant products are fruits.",
    "The concept 'vegetable' is somewhat fuzzy in its extension.",

    # === Geography / Entities ===
    "Paris is a specific city that instantiates the abstract concept 'capital'.",
    "Mount Everest is an instance of the class 'mountain'.",
    "Countries form a superset that contains France, Brazil, and Japan as elements.",
    "The abstract notion 'continent' has seven members in the common modern classification.",

    # === More hasExtension / intension-extension language ===
    "What is the extension of the concept 'unicorn'? Currently the empty set.",
    "The intension of 'water' is H2O; its extension is every actual body of water on Earth.",
    "The abstract category 'prime minister' has a small extension at any given time.",

    # === Negative / contrast examples ===
    "A cat is not a dog. Dogs and cats are both mammals but belong to different species.",
    "A car is not an animal. The category animal does not include artifacts.",
    "The number 4 is even, but the concept 'even number' is not itself a number.",
]

def collect_activations_multi_layer(model, hooks: list[str], n_tokens_per_layer: int = 200_000, batch_size: int = 16):
    """
    Aggressive collection: mix curated ontological prompts with general text for better features.
    Uses a small slice of pile-10k for diversity if available.
    Returns dict: hook -> (acts_tensor, labels_list)
    """
    full_texts = []
    target = n_tokens_per_layer * 4

    # Start with many repeats of our strong signal prompts
    while len(full_texts) * CONTEXT < target // 2:
        full_texts.extend(PROMPTS)

    # Add general text for natural distribution (more aggressive for real features)
    try:
        from datasets import load_dataset
        print("  Mixing in general text from pile-10k for better feature quality...")
        ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
        for ex in ds.take(2000):
            text = ex.get("text", "")[:500]
            if len(text) > 20:
                full_texts.append(text)
                if len(full_texts) * CONTEXT > target:
                    break
    except Exception as e:
        print(f"  Could not load pile (using only curated): {e}")
        # Fallback: repeat prompts more
        while len(full_texts) * CONTEXT < target:
            full_texts.extend(PROMPTS)

    print(f"  Collecting up to ~{n_tokens_per_layer} tokens per layer for hooks: {hooks} (total texts: {len(full_texts)})")

    collected = {h: [] for h in hooks}
    labels = {h: [] for h in hooks}

    num_batches = (len(full_texts) + batch_size - 1) // batch_size
    print_every = max(1, num_batches // 20)  # print ~20 progress updates

    for bi, i in enumerate(range(0, len(full_texts), batch_size)):
        batch = full_texts[i:i+batch_size]
        if not batch: break

        with torch.no_grad():
            _, cache = model.run_with_cache(batch, names_filter=hooks, return_type=None)

        for hook in hooks:
            acts = cache[hook]
            flat = acts.reshape(-1, acts.shape[-1]).cpu()
            collected[hook].append(flat)
            for txt in batch:
                labels[hook].extend([txt[:200]] * acts.shape[1])

        if (bi + 1) % print_every == 0 or bi == num_batches - 1:
            total_so_far = sum(a.shape[0] for a in collected[hooks[0]])
            print(f"    progress: batch {bi+1}/{num_batches}, ~{total_so_far} tokens for first hook")

        total = sum(a.shape[0] for a in collected[hooks[0]])
        if total >= n_tokens_per_layer:
            break

    results = {}
    for hook in hooks:
        acts = torch.cat(collected[hook], dim=0)[:n_tokens_per_layer]
        lbls = labels[hook][:n_tokens_per_layer]
        results[hook] = (acts, lbls)
        print(f"    {hook}: collected {acts.shape[0]} activation vectors")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama-3.1-8b", help="Short name or full HF id")
    parser.add_argument("--layers", default="12,16", help="Comma separated layer indices for resid_post")
    parser.add_argument("--tokens", type=int, default=TRAINING_TOKENS_TARGET)
    args = parser.parse_args()

    layer_list = [int(x) for x in args.layers.split(",")]

    print("=" * 70)
    print("PHASE 1 — TARGET MODEL RUN (Llama-3.1-8B / Mistral-7B class)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Layers: {layer_list}")
    print(f"Target tokens per layer: ~{args.tokens}")
    print("IMPORTANT: If this fails with 401/gated, run `huggingface-cli login` first.")
    print("=" * 70)

    try:
        model = load_model(args.model, device="auto")
    except Exception as e:
        err = str(e)
        print("Model load error:", type(e).__name__, err[:300])
        print("\n" + "!" * 70)
        print("GATED ACCESS OR MODEL SUPPORT ISSUE")
        print("!" * 70)
        print("For the primary targets (Llama-3.1-8B-Instruct or Mistral-7B):")
        print("1. Make sure you have accepted the license on the HF model page.")
        print("2. Run in your shell:   huggingface-cli login")
        print("   (or export HF_TOKEN=...)")
        print("3. You may also need a newer version of transformer-lens that lists these models.")
        print("   Current workaround: use full name after login, or try Llama-3.2 variants if open.")
        print("!" * 70)
        sys.exit(1)

    d_model = model.cfg.d_model
    n_layers = model.cfg.n_layers
    print(f"\nModel ready. d_model={d_model}, n_layers={n_layers}")

    hooks = [f"blocks.{i}.hook_resid_post" for i in layer_list if i < n_layers]

    # === Aggressive multi-layer collection (one forward pass for all hooks) ===
    print("\n=== Collecting activations for all requested layers ===")
    layer_data = collect_activations_multi_layer(model, hooks, n_tokens_per_layer=args.tokens, batch_size=BATCH_SIZE)

    all_saes = {}
    all_summaries = {}

    for hook in hooks:
        print(f"\n=== Processing {hook} ===")
        acts, labels = layer_data[hook]

        d_sae = int(d_model * D_SAE_EXPANSION)
        print(f"Training BatchTopK SAE (d_sae={d_sae}, expansion={D_SAE_EXPANSION}) ...")
        sae = train_sae_on_activations(
            activations=acts,
            d_model=d_model,
            d_sae=d_sae,
            lr=1.5e-4,
            steps=1500,           # max aggression: more steps for larger dataset
            batch_size=256,
            device="cpu",         # MPS float32/16 still flaky for very large collection + training
            seed=42,
        )
        all_saes[hook] = sae

        # Analysis
        print("Analyzing top features...")
        sample_size = min(8192, acts.shape[0])
        feat_acts = compute_feature_activations(sae, acts[:sample_size], batch_size=128)
        top = find_top_examples_per_feature(feat_acts, labels[:sample_size], top_k=6, min_activation=0.5)
        summaries = summarize_features(feat_acts, top)
        all_summaries[hook] = summaries

        safe_name = args.model.replace("/", "_").replace("-", "_")
        out_dir = Path(f"experiments/first_set/sae_runs/{safe_name}_layer{hook.split('.')[1]}")
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "model": args.model,
            "hook": hook,
            "d_model": d_model,
            "d_sae": d_sae,
            "tokens": acts.shape[0],
            "expansion": D_SAE_EXPANSION,
        }
        save_feature_catalog(summaries, out_dir / "feature_catalog.json", meta)

        # Persist the trained SAE for later consistency / circuit work
        try:
            sae_path = out_dir / "sae.pt"
            torch.save({"state_dict": sae.state_dict(), "config": {"d_in": d_model, "d_sae": d_sae}}, sae_path)
            print(f"  Saved SAE weights to {sae_path}")
        except Exception as e:
            print(f"  Could not save SAE weights: {e}")

        print(f"Saved catalog to {out_dir}")

    # === Basic consistency / LTC-style analysis ===
    print("\n=== Basic cross-layer consistency analysis ===")
    # Look for features that activate on similar "ontological" sentences across layers
    key_sentences = [
        "The concept of 'dog' has an extension",
        "Fido is an instance of dog",
        "All birds are animals, but not all animals are birds",
        "An abstract idea like 'justice' is different from the set",
        "Mammals are a superset of dogs and cats",
    ]

    for sent in key_sentences:
        print(f"\nSentence: {sent[:70]}...")
        for hook in hooks:
            summaries = all_summaries[hook]
            # Find top features that have this sentence (or very similar) in top examples
            hits = []
            for s in summaries:
                for ex in s.top_examples:
                    if sent.split()[0] in ex.text and sent.split()[-1][:5] in ex.text:
                        hits.append((s.feature_idx, s.max_activation))
                        break
            if hits:
                hits.sort(key=lambda x: -x[1])
                print(f"  {hook}: top hits {hits[:3]}")

    print("\n=== Run complete (maximum aggression mode) ===")
    print("Review the per-layer catalogs. Consider scaling tokens further or adding more layers.")

if __name__ == "__main__":
    main()
