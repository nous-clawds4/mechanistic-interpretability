# SAE Training Prompts & Guidelines (Phase 1)

These are reusable instructions / notes for training SAEs in this project.

## Core Objective (Phase 1)
Discover features in residual stream activations (esp. layers 8-20 of 7-8B models) that appear to represent:
- Abstract category concepts (intensions)
- Specific instances / entities (members of categories)
- Signals related to hierarchy or set membership

## Recommended Starting Hyperparameters (Llama-3.1-8B class)
- Hook points: blocks.{8..20}.hook_resid_post (start with one or two)
- d_sae: 24576 (expansion ~6x) or 16384 for faster first passes
- context_size: 128 (conservative for Mac memory)
- batch_size: 2-8 (MPS/CPU limited)
- l1_coefficient: 3e-3 to 1e-2 (tune for ~1-5% sparsity on average)
- lr: 3e-5 to 7e-5
- Training tokens: start with 1M-5M tokens for initial discovery runs

## Data
- Good starter: NeelNanda/pile-10k (or larger filtered subsets)
- Prefer natural text containing category language, lists, "is a kind of", definitions, etc.
- Later: targeted datasets mixing category names, instances, and relational sentences.

## Evaluation of SAE Quality (Phase 1)
1. Reconstruction loss vs L0 / sparsity curves
2. Manual review of top-activating examples for a sample of features
3. Rough auto-interp using an LLM to describe "what this feature fires on"
4. Look for features that cleanly separate abstract vs concrete language

## Output Artifacts (per run)
- run_config.yaml (exact params)
- sae weights / state
- feature_catalog.json + .md (top examples)
- short written summary in the notebook

## Risks specific to this hardware
- MPS support in sae_lens / torch is partial. Expect some CPU fallback.
- Keep activation caching modest. Use `context_size` and `batch_size` small initially.
- Watch for NaNs in training; lower lr or increase l1 if unstable.

See `EXPERIMENTS.md` and the Phase 1 notebook for concrete steps.
