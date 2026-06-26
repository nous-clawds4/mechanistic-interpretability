#!/usr/bin/env python3
"""
Phase 1 — Small gpt2 SAE experiment (robust version).

Collects real residual stream activations from gpt2 using transformer_lens.
Trains a SAE using the lower-level reliable path (BatchTopK or our fallback).
Produces a feature catalog with analysis on curated ontological-style prompts.

This version avoids fragile high-level runner save paths.
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.model_loading import load_model
from src.sae.train_sae import train_sae_on_activations
from src.sae.analyze_features import (
    compute_feature_activations,
    find_top_examples_per_feature,
    summarize_features,
    save_feature_catalog,
    filter_for_ontological_interest,
)

# ----------------------------- Config -----------------------------
MODEL_NAME = "gpt2"
HOOK = "blocks.6.hook_resid_post"
D_SAE = 6144          # 8x for gpt2 (768)
N_SAMPLES = 4096      # number of activation vectors
CONTEXT = 32
SEED = 42
OUTPUT_DIR = Path("experiments/first_set/sae_runs/gpt2_layer6_small")

# A mix of prompts designed to probe categories / instances / relations
ONTOLOGY_PROMPTS = [
    "A dog is a mammal.",
    "Mammals include dogs, cats, and whales.",
    "Fido is a specific dog.",
    "Cars, trucks and airplanes are all vehicles.",
    "A car is a type of vehicle.",
    "Berlin is a city in Germany.",
    "An apple is a fruit.",
    "All birds can fly, except for some like penguins.",
    "The concept of justice is abstract.",
    "Earth is an instance of a planet.",
    "A poodle is a kind of dog.",
    "Superset of 'dog' includes 'mammal'.",
    "Specific example: the Eiffel Tower is a landmark.",
    "Fruits are a category that contains apples and oranges.",
]

def collect_activations(model, hook_name: str, n_vecs: int = 4096, context: int = 32):
    """Collect flattened residual activations from the model on mixed data."""
    torch.manual_seed(SEED)

    # Use the ontology prompts + some generic text to get variety
    base_texts = ONTOLOGY_PROMPTS * 20
    # Add some generic text
    generic = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "To be or not to be, that is the question.",
        "It was the best of times, it was the worst of times.",
    ] * 10
    texts = (base_texts + generic)[:128]

    _, cache = model.run_with_cache(
        texts,
        names_filter=[hook_name],
        return_type=None,
    )
    acts = cache[hook_name]  # [batch, seq, d_model]
    flat = acts.reshape(-1, acts.shape[-1])

    # Take a random subset to keep memory reasonable
    if flat.shape[0] > n_vecs:
        idx = torch.randperm(flat.shape[0])[:n_vecs]
        flat = flat[idx]

    # Build corresponding text labels (approximate — whole prompt for each vec)
    labels = []
    for t in texts:
        labels.extend([t] * min(context, acts.shape[1]))
    labels = labels[: flat.shape[0]]

    return flat.cpu(), labels


def main():
    print("=" * 62)
    print("PHASE 1 — REAL SMALL RUN ON gpt2 (robust activation pipeline)")
    print("=" * 62)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model: {MODEL_NAME}")
    model = load_model(MODEL_NAME, device="cpu")

    d_model = model.cfg.d_model
    print(f"  d_model = {d_model} | using d_sae = {D_SAE}")

    print("\n[1] Collecting real activations from curated + generic text...")
    acts, labels = collect_activations(model, HOOK, n_vecs=N_SAMPLES, context=CONTEXT)
    print(f"    Collected {acts.shape[0]} activation vectors of dim {acts.shape[1]}")

    print("\n[2] Training SAE (BatchTopK via sae_lens if available, else fallback)...")
    sae = train_sae_on_activations(
        activations=acts,
        d_model=d_model,
        d_sae=D_SAE,
        lr=3e-4,
        l1_coefficient=0.003,
        steps=800,
        batch_size=256,
        device="cpu",
        seed=SEED,
    )
    print(f"    SAE trained: {type(sae)}")

    print("\n[3] Analyzing features on ontology-flavored prompts...")
    # Re-collect a small set focused on the test prompts for clean analysis
    test_acts, test_labels = collect_activations(
        model, HOOK, n_vecs=1024, context=CONTEXT
    )

    feat_acts = compute_feature_activations(sae, test_acts, batch_size=128)
    top_ex = find_top_examples_per_feature(feat_acts, test_labels, top_k=4, min_activation=0.2)
    summaries = summarize_features(feat_acts, top_ex)

    catalog_path = OUTPUT_DIR / "feature_catalog.json"
    save_feature_catalog(
        summaries,
        catalog_path,
        model_info={
            "model": MODEL_NAME,
            "hook": HOOK,
            "d_model": d_model,
            "d_sae": D_SAE,
            "n_activations_trained": acts.shape[0],
            "note": "Small gpt2 run — proof of pipeline. Features are preliminary.",
        },
    )

    interesting = filter_for_ontological_interest(
        summaries, min_max_act=0.8, max_sparsity=0.3
    )

    print("\n" + "-" * 62)
    print("SUMMARY OF FINDINGS (gpt2 layer 6)")
    print("-" * 62)
    print(f"Features analyzed on test set : {len(summaries)}")
    print(f"Passed rough 'ontological' filter: {len(interesting)}")
    print(f"Catalog saved to: {catalog_path}")
    print(f"Human-readable:   {catalog_path.with_suffix('.md')}")

    print("\nHighest max-activation features (sample):")
    for s in sorted(summaries, key=lambda x: -x.max_activation)[:6]:
        ex_text = s.top_examples[0].text[:85].replace("\n", " ") if s.top_examples else ""
        print(f"  {s.feature_idx:4d}  max={s.max_activation:7.2f}  sparsity={s.sparsity:.3f}  | {ex_text}")

    if interesting:
        print("\nFeatures passing filter (possible category/instance signals):")
        for s in sorted(interesting, key=lambda x: -x.max_activation)[:5]:
            ex = s.top_examples[0].text[:80] if s.top_examples else ""
            print(f"  feat {s.feature_idx}: max={s.max_activation:.2f} — {ex}")
    else:
        print("\nNo features passed the current strict filter (common on small runs / small models).")

    print("\nThis run demonstrates the complete Phase 1 pipeline:")
    print("  model load → activation collection → SAE training → top-example analysis → catalog")
    print("Next: scale to Llama-3.1-8B (after HF login) and larger data.")
    print("=" * 62)


if __name__ == "__main__":
    main()
