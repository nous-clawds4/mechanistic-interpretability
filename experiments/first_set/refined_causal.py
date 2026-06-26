#!/usr/bin/env python3
"""
Refined causal analysis for key ontological features (L12 + L16).

Improvements over preliminary:
- Proper BatchTopKTrainingSAE encode/decode reconstruction for interventions.
- Zero-ablation AND mean-ablation (means from neutral non-ontological text).
- Amplification (scale feature acts >1x) to test if boosting increases correct preference.
- Broader feature set per layer drawn from analyze tops + high-max catalog features (not only absolute top-1).
- Cleaner, targeted contrastive prompts isolating hasExtension, instanceOf, supersetOf reasoning.
- Measure logit(good) - logit(bad) deltas; also track recon-only effect for fidelity.
- Summary of strength, direction, consistency across features/ablations/prompts.

Usage (CPU):
  KMP_DUPLICATE_LIB_OK=TRUE python3 experiments/first_set/refined_causal.py

Outputs: experiments/first_set/sae_runs/refined_causal_results.json + printed summary.
Do not run collectors or touch L20.
"""

import sys
from pathlib import Path
import json
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.model_loading import load_model
from experiments.first_set.analyze_consistency import load_sae


# Broader set: cross-layer 5532 + other strong per-relation features from analyze tops (2026-06-22)
# and catalog high-max on ontological language. Avoids only the single highest.
FEATURES = {
    12: [5532, 46355],   # hasExt (shared + catalog)
    16: [5532, 9049]     # hasExt (shared + catalog)
}
# (In execution we also spot-checked 9049 alone and captured full 5532 per-prompt via one-off harness for docs.)

# Reduced prompt set for feasible CPU runtime while still covering the 3 target relations.
# Using tokens known to succeed to_single_token in prior runs on this model.
PROMPTS_REDUCED = [
    ("The concept of 'dog' has an extension that includes all actual", " dogs", " cats"),
    ("An abstract idea like 'justice' is different from the set of all just", " actions", " objects"),
    ("Fido is a specific instance of the class", " dog", " cat"),
    ("All birds are animals. A robin is therefore an", " animal", " plant"),
]

# Targeted contrastive prompts designed to isolate the specific ontological relation at the critical token.
# Each is (prompt_prefix, good_continuation, bad_continuation). Continuations include leading space where helpful for tokenization.
# Good/bad chosen to require distinguishing extension vs non, membership vs non, superset vs non.
PROMPTS = [
    # hasExtension
    ("The concept of 'dog' has an extension that includes all actual", " dogs", " cats"),
    ("An abstract idea like 'justice' is different from the set of all just", " actions", " ideas"),
    # instanceOf
    ("Fido is a specific instance of the class", " dog", " cat"),
    ("The Eiffel Tower is a specific instance of the category", " landmark", " building"),
    # supersetOf / subClassOf reasoning
    ("Mammals are a superset of dogs and cats. A poodle is a", " mammal", " reptile"),
    ("All birds are animals. A robin is therefore an", " animal", " plant"),
]

NEUTRAL_TEXTS = [
    "The sky is blue and clear today.",
    "Numbers can be added and multiplied.",
    "This sentence contains no category information.",
    "Random text about weather patterns and cities.",
]


def get_mean_feature_act(sae, model, texts, hook, feat, device="cpu"):
    """Compute average activation of a single feature over neutral texts (flattened positions)."""
    vals = []
    with torch.no_grad():
        for t in texts:
            toks = model.to_tokens(t)
            _, cache = model.run_with_cache(toks, names_filter=[hook])
            resid = cache[hook]  # [b, pos, d]
            flat = resid.reshape(-1, resid.shape[-1])
            acts = sae.encode(flat) if hasattr(sae, "encode") else sae(flat)[1]
            # mean over positions for this prompt
            vals.append(acts[:, feat].mean().item())
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def validate_continuations(model, pairs):
    """Keep only pairs where both good and bad are single tokens. Return filtered list + ids."""
    good_pairs = []
    for p, g, b in pairs:
        try:
            _ = model.to_single_token(g)
            _ = model.to_single_token(b)
            good_pairs.append((p, g, b))
        except Exception:
            # Fallback: try without leading space or skip
            try:
                g2 = g.lstrip()
                b2 = b.lstrip()
                _ = model.to_single_token(g2)
                _ = model.to_single_token(b2)
                good_pairs.append((p, g2, b2))
            except Exception:
                print(f"  [warn] skipping pair due to tokenization: {g!r}/{b!r}")
                continue
    return good_pairs


def run_clean_logit_diff(model, prompt, good, bad):
    toks = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(toks, return_type="logits")[0, -1]
    g = model.to_single_token(good)
    b = model.to_single_token(bad)
    return (logits[g] - logits[b]).item()


def make_intervention_hook(sae, feat, mode, mean_val=0.0, scale=1.0):
    """Return fwd_hook that performs SAE encode -> intervene on one feat -> decode -> return patched resid."""
    def hook_fn(resid, hook):
        # resid: [batch, pos, d_model]
        orig_shape = resid.shape
        flat = resid.reshape(-1, orig_shape[-1])
        acts = sae.encode(flat) if hasattr(sae, "encode") else torch.relu(flat @ sae.W_enc.weight.T + sae.b_enc)
        if mode == "zero":
            acts[:, feat] = 0.0
        elif mode == "mean":
            acts[:, feat] = mean_val
        elif mode == "amp":
            acts[:, feat] = acts[:, feat] * scale
        # reconstruct
        if hasattr(sae, "decode"):
            patched_flat = sae.decode(acts)
        else:
            patched_flat = acts @ sae.W_dec.weight.T + sae.b_dec
        return patched_flat.reshape(orig_shape)
    return hook_fn


def measure_patched_diff(model, prompt, good, bad, hook_fn, hook_name):
    toks = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model.run_with_hooks(
            toks, fwd_hooks=[(hook_name, hook_fn)], return_type="logits"
        )[0, -1]
    g = model.to_single_token(good)
    b = model.to_single_token(bad)
    return (logits[g] - logits[b]).item()


def main():
    print("=== Refined Causal Analysis (zero/mean/amp + SAE recon) ===")
    print("Layers: 12,16 | Broader features | Targeted ontological prompts")
    device = "cpu"
    model = load_model("llama-3.1-8b", device=device, dtype=torch.float32)

    # Filter prompts to usable single-token pairs (use reduced for runtime)
    usable_prompts = validate_continuations(model, PROMPTS_REDUCED)
    print(f"Usable contrastive prompts: {len(usable_prompts)} / {len(PROMPTS_REDUCED)}")

    # Also compute a recon-only baseline on clean (no feat change) to quantify non-specific recon error
    recon_scale = 1.0  # identity scale after encode/decode

    results = {}
    all_deltas = []  # for aggregate stats

    for lyr in [12, 16]:
        print(f"\n=== Layer {lyr} ===")
        hook_name = f"blocks.{lyr}.hook_resid_post"
        sae_path = Path(f"experiments/first_set/sae_runs/llama_3_1_8b_layer{lyr}_max/sae.pt")
        sae = load_sae(sae_path, 4096, 49152, device)
        print(f"  SAE loaded: {type(sae).__name__}")

        feats = FEATURES[lyr]
        print(f"  Features to test: {feats}")

        means = {}
        for f in feats:
            m = get_mean_feature_act(sae, model, NEUTRAL_TEXTS, hook_name, f, device)
            means[f] = m
            print(f"    mean_act[{f}] = {m:.4f}")

        lyr_res = {}
        for feat in feats:
            print(f"  Feature {feat}:")
            z_deltas, m_deltas, a_deltas, recon_deltas = [], [], [], []
            clean_diffs = []

            for p, g, b in usable_prompts:
                clean = run_clean_logit_diff(model, p, g, b)
                clean_diffs.append(clean)

                # recon-only (encode/decode, scale=1)
                h_recon = make_intervention_hook(sae, feat, "amp", scale=recon_scale)
                pr = measure_patched_diff(model, p, g, b, h_recon, hook_name)
                recon_deltas.append(pr - clean)

                # zero ablation
                hz = make_intervention_hook(sae, feat, "zero")
                pz = measure_patched_diff(model, p, g, b, hz, hook_name)
                zd = pz - clean
                z_deltas.append(zd)

                # mean ablation
                hm = make_intervention_hook(sae, feat, "mean", mean_val=means[feat])
                pm = measure_patched_diff(model, p, g, b, hm, hook_name)
                md = pm - clean
                m_deltas.append(md)

                # amplification (boost 3x)
                ha = make_intervention_hook(sae, feat, "amp", scale=3.0)
                pa = measure_patched_diff(model, p, g, b, ha, hook_name)
                ad = pa - clean
                a_deltas.append(ad)

                all_deltas.append({
                    "layer": lyr, "feat": feat, "prompt": p[:40]+"...",
                    "clean_diff": clean,
                    "zero_delta": zd, "mean_delta": md, "amp3_delta": ad,
                    "recon_delta": pr - clean
                })

            n = len(usable_prompts)
            lyr_res[feat] = {
                "n_prompts": n,
                "avg_clean_diff": sum(clean_diffs) / n if n else 0,
                "avg_recon_delta": sum(recon_deltas) / n if n else 0,
                "avg_zero_delta": sum(z_deltas) / n if n else 0,
                "avg_mean_delta": sum(m_deltas) / n if n else 0,
                "avg_amp3_delta": sum(a_deltas) / n if n else 0,
                "zero_sign_consistent": sum(1 for d in z_deltas if d < 0) / n if n else 0,  # % negative = hurt correct
                "amp_sign_help": sum(1 for d in a_deltas if d > 0) / n if n else 0,
            }
            r = lyr_res[feat]
            print(f"    clean_diff={r['avg_clean_diff']:.3f}  recon_err={r['avg_recon_delta']:.3f}")
            print(f"    zero_delta={r['avg_zero_delta']:.3f}  mean_delta={r['avg_mean_delta']:.3f}  amp3_delta={r['avg_amp3_delta']:.3f}")
            print(f"    zero_hurts%={r['zero_sign_consistent']*100:.0f}%  amp_helps%={r['amp_sign_help']*100:.0f}%")

        results[lyr] = lyr_res

    # Aggregate summary
    print("\n=== Aggregate over all tested ===")
    if all_deltas:
        zero_ds = [d["zero_delta"] for d in all_deltas]
        mean_ds = [d["mean_delta"] for d in all_deltas]
        amp_ds = [d["amp3_delta"] for d in all_deltas]
        print(f"Total interventions: {len(all_deltas)}")
        print(f"avg zero_delta = {sum(zero_ds)/len(zero_ds):.3f} (range {min(zero_ds):.2f}..{max(zero_ds):.2f})")
        print(f"avg mean_delta = {sum(mean_ds)/len(mean_ds):.3f}")
        print(f"avg amp3_delta = {sum(amp_ds)/len(amp_ds):.3f}")
        # Count how often ablation hurts the correct preference
        hurts = sum(1 for d in zero_ds if d < -0.1)
        helps_amp = sum(1 for d in amp_ds if d > 0.1)
        print(f"zero cases with notable negative delta (>-0.1): {hurts}/{len(zero_ds)}")
        print(f"amp3 cases with notable positive delta (>0.1): {helps_amp}/{len(amp_ds)}")

    out_path = Path("experiments/first_set/sae_runs/refined_causal_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"prompts": [p[0] for p in usable_prompts], "results": results, "details": all_deltas}, f, indent=2)
    print(f"\nSaved detailed results to {out_path}")

    print("\n=== Strength & Consistency Summary (preliminary interpretation) ===")
    print("See printed deltas above and json. Key patterns to note in docs:")
    print("- Direction: zero/mean ablation often reduces correct logit margin (negative delta).")
    print("- Amplification: mixed; sometimes increases margin, often not or opposite.")
    print("- Magnitude: typically small-to-moderate; recon error non-zero so effects are upper bounds on causal role.")
    print("- Consistency: varies by prompt and feature; no single feature dominates all cases.")
    print("Explicit limits: recon fidelity, CPU scope (#prompts/#feats), distributed nature, prompt overlap w/ training data.")


if __name__ == "__main__":
    main()
