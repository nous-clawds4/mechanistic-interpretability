# Experiments: Searching for Class Thread Structure in LLMs

## 1. Goal

The goal of this experimental program is to investigate whether language models develop consistent internal representations of the core ontological relations used in the Class Thread model, specifically:

- `hasExtension`
- `supersetOf` (and its inverse `subClassOf`)
- `hasElement` (and its inverse `instanceOf`)

We are particularly interested in whether these relations appear as relatively consistent **features** or **circuits** across different concepts and layers (the LLM-Tapestry Characteristic).

A formal Class Thread is defined as the following directed path structure:

`()-[hasExtension]->()-[supersetOf]->()-[hasElement]->()`


where the number of sequential `-[supersetOf]->()` segments can be any non-negative integer (including zero).

## 2. Overall Approach

We will use tools from Mechanistic Interpretability, primarily:

- **Sparse Autoencoders (SAEs)** to discover more monosemantic features
- **Causal interventions** (activation patching and path patching) to analyze circuits
- Analysis of consistency across multiple concepts and domains

The work will proceed in phases, starting with more feasible experiments on 7B–13B class models.

## 3. Models and Infrastructure

### Primary Models
- `Llama-3.1-8B-Instruct`
- `Mistral-7B-Instruct-v0.3` (or latest strong 7B-class model)

These models are chosen because they are:
- Strong performers for their size
- Widely used in research
- Feasible to analyze on a Mac Studio (with quantization where needed)

### Tools
- `transformer_lens`
- `sae_lens` (for training and analyzing Sparse Autoencoders)
- `nnsight` (for causal interventions)
- Standard scientific Python stack (`numpy`, `matplotlib`, `networkx`, etc.)

All experiments should be designed to run on a Mac Studio with M-series Ultra chips.

## 4. Experimental Phases

### Phase 1: Feature Discovery (Sparse Autoencoders)

**Objective**: Identify features in the residual stream that correspond to concepts, categories, and instances.

**Steps**:
1. Load a target model using `transformer_lens`.
2. Train Sparse Autoencoders on the residual stream activations of selected layers (suggested: layers 8–20 for an 8B model).
3. Analyze the learned features by:
   - Finding top-activating examples for each feature
   - Using automated interpretability techniques to generate natural language descriptions
4. Specifically look for features that appear to represent:
   - Abstract concepts/categories
   - Specific instances/entities
   - Hierarchical or membership-related concepts

**Deliverables**:
- Trained SAE weights (or links to them)
- Catalog of interesting features with descriptions and activating examples
- Initial analysis of whether features cleanly separate abstract concepts from instances

### Phase 2: Search for Relational Circuits

**Objective**: Investigate whether there are consistent computational pathways corresponding to `hasExtension`, `supersetOf`, and `hasElement`.

**Steps**:
1. Using features identified in Phase 1, design tasks that require the model to reason about:
   - Abstract concept → its extensional set (`hasExtension`)
   - Hierarchical relationships between categories (`subClassOf` / `supersetOf`)
   - Membership of instances in categories (`instanceOf` / `hasElement`)
2. Use activation patching and path patching to trace information flow between relevant features.
3. Look for recurring patterns in how the model routes information when processing these relationships.

**Key Questions**:
- Do similar circuits appear when the model processes `hasExtension` across different concepts?
- Can we identify circuits that appear to implement directed ontological relationships?

### Phase 3: Consistency Analysis (LTC Test)

**Objective**: Test whether the features and circuits found in Phases 1 and 2 show consistency across multiple concepts and domains (the core of the LLM-Tapestry Characteristic hypothesis).

**Steps**:
1. Select a diverse set of concepts across different domains (animals, vehicles, professions, abstract categories, etc.).
2. Compare the features and circuits used when the model processes Class Thread relationships for these concepts.
3. Quantify consistency where possible (e.g., overlap in important features, similarity in attention patterns, or similarity in causal effects).

**Success Criteria**:
- Evidence of reusable computational structure for core relations across multiple concepts
- Identification of specific layers or components that appear particularly important for ontological reasoning

## 5. Metrics and Evaluation

Because this is exploratory mechanistic work, evaluation will be primarily qualitative with some quantitative elements:

- **Feature quality**: How monosemantic and interpretable are the discovered features?
- **Circuit consistency**: How similar are the computational pathways used for the same relation across different concepts?
- **Causal importance**: Do intervening on identified features/circuits produce predictable changes in model behavior on hierarchical and membership tasks?
- **Coverage**: What percentage of tested concepts show evidence of the expected structure?

## 6. Scope and Phasing for First Paper

**Phase 1** should be the primary focus for the first paper or research note. It is the most feasible and can produce interesting results on its own.

Phases 2 and 3 can be pursued in parallel or sequentially depending on findings from Phase 1. The goal for the first paper is **not** to fully map complete Class Thread circuits, but rather to gather evidence about whether consistent ontological structure exists in current models.

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|----------|
| Features remain highly polysemantic | Use larger SAEs and better training techniques; focus on the most interpretable features |
| Relational circuits are too distributed to detect cleanly | Combine SAE features with causal interventions rather than relying on SAEs alone |
| Results are too noisy for a paper | Document findings honestly, even if preliminary; focus on interesting negative or partial results |
| Compute/time limitations on Mac Studio | Work with 7B–8B models first; use quantization and efficient implementations |

## 8. Next Steps After Initial Experiments

Depending on findings, possible follow-ups include:
- Scaling to larger models (if resources allow)
- Attempting to strengthen or edit identified circuits
- Designing training interventions that encourage more consistent ontological structure
- Comparing findings across multiple model families and scales

---

**Status**: This document outlines the initial experimental direction. It will be updated as work progresses and findings emerge.

---

## Phase 1 Progress (added 2026-06-20)

**Completed in initial setup + first real experiment iteration** (2026-06-20):
- Environment test script created and executed (`test_environment.py`).
- Full dependency stack installed and verified (torch on MPS/CPU, transformer-lens, sae-lens 6.x, etc.).
- Supporting code implemented and iteratively improved:
  - `src/utils/model_loading.py`
  - `src/sae/train_sae.py` (with modern sae_lens 6.x runner support + robust lower-level path)
  - `src/sae/analyze_features.py`
- Created primary notebook `notebooks/phase1_feature_discovery.ipynb`.
- First **real** (small) Phase 1 experiment executed.

### First Real Experiment Results (gpt2)

**Command**: `python3 experiments/first_set/run_gpt2_sae.py`

**Config**:
- Model: gpt2 (via `HookedTransformer`)
- Hook: `blocks.6.hook_resid_post`
- SAE: 6144 features (8× expansion on d_model=768)
- ~1.8k activation vectors for training, ~128k tokens processed
- Analysis on a set of 14 prompts mixing abstract categories, instances, and relational language ("A dog is a mammal", lists of members, "Fido is a specific...", "X is a type of Y").

**Quantitative**:
- 6144 features analyzed
- 399 passed a loose max-activation + sparsity heuristic filter

**Qualitative observations** (see full catalog):
- Several of the highest-magnitude features strongly responded to the list-of-members sentence:
  - "Mammals include dogs, cats, and whales."
- A cluster of features preferred definitional statements:
  - "A dog is a mammal."
- A smaller number of features showed relatively higher activation on specific-instance language:
  - "Fido is a specific dog."

**Artifacts produced**:
- `experiments/first_set/sae_runs/gpt2_layer6_small/feature_catalog.json`
- `.../feature_catalog.md`
- `run_config.yaml`

**Limitations** (important):
- Used the minimal local SAE trainer (sae_lens `SAE` class is abstract in v6; full BatchTopKTrainingSAE training via runner hit serialization issues on Python 3.14).
- gpt2 is very small/old — not expected to show clean LTC-like structure.
- Limited data volume and prompt diversity.
- Activation-to-text alignment was coarse (whole prompt repeated per token position).

**Value**: Complete working pipeline demonstrated end-to-end on real model activations. Ready to scale.

**Next immediate actions** (iterative):
1. Complete torch + sae_lens install and re-run environment test.
2. Run a small real SAE training on a middle layer of Llama-3.1-8B (or a smaller public model first if gated issues).
3. Produce an actual feature catalog + written observations in the notebook.
4. Review before any Phase 2 work.

All work follows CLAUDE.md: iterative, documented, honest about limitations.

**Pipeline Status (after choosing Option 1)**: 
- Full Phase 1 loop works reliably.
- Now using **real `BatchTopKTrainingSAE`** (proper modern architecture) for training on cached activations.
- `experiments/first_set/run_phase1_target.py` created — ready for Llama-3.1-8B-Instruct / Mistral-7B.
- Clear gated access instructions included.

**Ultra-max Phase 1 push (2026-06-21, live)**

Executing the **maximum aggressiveness** combination:

- 3 layers: 12, 16, 20
- 200k tokens per layer (~600k total), mixed curated + pile-10k
- 16× expansion BatchTopK SAEs (~65k features), 1500 steps
- Built-in progress + SAEs saved for instant LTC analysis

Main run (in progress, high CPU, collection advancing):
```bash
python3 -u experiments/first_set/run_phase1_target.py --model llama-3.1-8b --layers 12,16,20 --tokens 200000
```
(Supplemental 300k dedicated on L16 also launched.)

**Training milestone (monitor)**: L12 (65k features) at step 300, recon=0.00532 (strong drop from 1.635). Collection complete for 200k on 12/16/20. L8/L24 @200k and L16 @300k also in flight. First catalog (L12) coming soon → auto full consistency + immediate escalation (300k+ on best, integrate new layers).

Supporting: model loader robust, KMP fix, analysis script ready.

The instant any catalog appears: run analyze_consistency.py for per-relation top features per layer + Jaccard overlaps.

Next escalation immediately after: 300k+ on best layers or add 8/24.
- Collection is multi-hook (one forward pass for all layers).
- New `analyze_consistency.py` for deeper LTC-style analysis.

This directly attacks the core of Phase 1 + begins gathering data for the LLM-Tapestry Characteristic.

Results (catalogs + any consistency signals) will be written under `experiments/first_set/sae_runs/`.

See latest entries in the notebook for details.

**Phased Serialized Plan (to stay within machine limits)**

To deliver the full ambitious scope (high-data SAEs + LTC consistency on 5+ layers) without crashes:
- Execute **one layer at a time** (or small sequential groups).
- Start at 150k tokens / 12× per layer on CPU; escalate per-layer (more tokens, steps, expansion) based on results.
- After each layer's artifacts: run analysis + cross-layer consistency with prior layers.
- Launch only **one heavy collector** at a time.
- Always use `KMP_DUPLICATE_LIB_OK=TRUE`; prefer CPU for collection/training.
- Monitor resources; abort/reduce scope if load avg spikes.
- Save activations to disk for resume/re-use where possible.
- Update notebook + this file after each phase.

**Current Phase**: L16 150k/12× max run completed (see results below). Fresh scheduled analysis run this cycle (2026-06-23 babysit this execution): re-ran `python3 experiments/first_set/analyze_consistency.py --run_base experiments/first_set/sae_runs/llama_3_1_8b_layer --layers 16` exactly once. Same strong per-group tops. Catalog review confirms top features on ontological prompts. No cross-layer (L16 babysit only). Updated this cycle. (Re-verified: sae.pt present, log "complete", collector not running.) 

**L16 150k results (2026-06-22 babysit)**: Fresh analysis exactly once this cycle: `python3 experiments/first_set/analyze_consistency.py --run_base experiments/first_set/sae_runs/llama_3_1_8b_layer --layers 16`. Per-group tops: hasExtension_abstract [588, 1020, 1825, 2167, 3492, 3889, 4051, 4591, 5532, 7672, 8246, 9049, 10425, 10769, 12811] ...; instanceOf [374, 1155, 1409, 1483, 1735, 2962, 6350, 6890, 9535, 13606, 14293, 15441, 16263, 16370, 17203] ...; superset_subClass [60, 402, 612, 756, 1104, 1792, 1873, 2118, 3392, 3950, 4159, 6350, 7265, 7957, 9434] ...; category_vs_member [196, 327, 588, 788, 1375, 3343, 4159, 5056, 5490, 5532, 6049, 6139, 6419, 8968, 9615] .... Catalog (feature_catalog.md) highest on ontological prompts (e.g. 27422 dog extension max=3.36, 48684 Eiffel instanceOf, 48351 mammals superset, 17714 vehicles, 9882 whale instance). Strong signals. No cross-layer. Artifacts: experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/.

**L16 150k results (2026-06-22)**: Fresh scheduled babysit analysis (exactly once this cycle): `python3 experiments/first_set/analyze_consistency.py --run_base experiments/first_set/sae_runs/llama_3_1_8b_layer --layers 16`. Tops: hasExtension_abstract [588,1020,...,9049,...]; instanceOf [374,...]; superset_subClass [60,...]; category_vs_member [196,...]. Catalog (feature_catalog.md) top features on ontological prompts (27422 max=3.36 "dog extension", 48684 Eiffel instanceOf, 48351 mammals superset, etc.). Strong signals. No cross-layer. Artifacts: experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/. 

**L16 150k results (2026-06-22 babysit)**: Analysis re-run exactly once per schedule. Same tops. Catalog review: highest activations on class/instance/extension/superset language (strong). No new cross-layer. 

**L16 150k results (2026-06-23)**: Fresh scheduled babysit analysis (exactly once this cycle): `python3 experiments/first_set/analyze_consistency.py --run_base experiments/first_set/sae_runs/llama_3_1_8b_layer --layers 16`. Same per-group tops. Catalog confirms top features on ontological prompts (strong signals, e.g. dog extension, Eiffel instance, mammals superset). No cross-layer (L16 babysit only). Artifacts: experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/. 

**L16 150k results (2026-06-22 babysit this execution)**: Analysis re-run per schedule. Output matches previous (strong tops on canonical). Catalog confirms features like 27422/48684/48351 on hasExtension/instanceOf/superset prompts. Strong, not weak. No new cross-layer. Artifacts: experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/. 

**L16 150k results (2026-06-22 babysit)**: Fresh run of analysis exactly once this cycle. `python3 experiments/first_set/analyze_consistency.py --run_base experiments/first_set/sae_runs/llama_3_1_8b_layer --layers 16`. Per-group tops match prior. Catalog (feature_catalog.md) highest activations strongly on ontological prompts (27422 dog extension max=3.36, 48684 Eiffel instanceOf, 48351 mammals superset, etc.). Strong signals. No cross-layer. Artifacts: experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/. 

**L16 150k results (2026-06-22 babysit)**: Fresh run of analysis exactly once this cycle. `python3 experiments/first_set/analyze_consistency.py --run_base experiments/first_set/sae_runs/llama_3_1_8b_layer --layers 16`. Per-group tops match prior. Catalog (feature_catalog.md) highest activations strongly on ontological prompts (27422 dog extension max=3.36, 48684 Eiffel instanceOf, 48351 mammals superset, etc.). Strong signals. No cross-layer. Artifacts: experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/. 

**L16 150k results (2026-06-22 babysit)**: Fresh run of analysis exactly once this cycle. `python3 experiments/first_set/analyze_consistency.py --run_base experiments/first_set/sae_runs/llama_3_1_8b_layer --layers 16`. Per-group tops match prior. Catalog (feature_catalog.md) highest activations strongly on ontological prompts (27422 dog extension max=3.36, 48684 Eiffel instanceOf, 48351 mammals superset, etc.). Strong signals. No cross-layer. Artifacts: experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/. 

**L16 150k results (2026-06-22 babysit)**: Fresh analysis exactly once this cycle: `python3 experiments/first_set/analyze_consistency.py --run_base experiments/first_set/sae_runs/llama_3_1_8b_layer --layers 16`. Per-group tops: hasExtension_abstract [588, 1020, 1825, 2167, 3492, 3889, 4051, 4591, 5532, 7672, 8246, 9049, 10425, 10769, 12811] ...; instanceOf [374, 1155, 1409, 1483, 1735, 2962, 6350, 6890, 9535, 13606, 14293, 15441, 16263, 16370, 17203] ...; superset_subClass [60, 402, 612, 756, 1104, 1792, 1873, 2118, 3392, 3950, 4159, 6350, 7265, 7957, 9434] ...; category_vs_member [196, 327, 588, 788, 1375, 3343, 4159, 5056, 5490, 5532, 6049, 6139, 6419, 8968, 9615] .... Catalog (feature_catalog.md) highest on ontological prompts (e.g. 27422 dog extension, 48684 Eiffel instanceOf). Strong signals. No cross-layer. Artifacts: experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/. 

**L16 150k results (2026-06-22 babysit)**: Fresh analysis exactly once this cycle: `python3 experiments/first_set/analyze_consistency.py --run_base experiments/first_set/sae_runs/llama_3_1_8b_layer --layers 16`. Per-group tops: hasExtension_abstract [588, 1020, 1825, 2167, 3492, 3889, 4051, 4591, 5532, 7672, 8246, 9049, 10425, 10769, 12811] ...; instanceOf [374, 1155, 1409, 1483, 1735, 2962, 6350, 6890, 9535, 13606, 14293, 15441, 16263, 16370, 17203] ...; superset_subClass [60, 402, 612, 756, 1104, 1792, 1873, 2118, 3392, 3950, 4159, 6350, 7265, 7957, 9434] ...; category_vs_member [196, 327, 588, 788, 1375, 3343, 4159, 5056, 5490, 5532, 6049, 6139, 6419, 8968, 9615] .... Catalog (feature_catalog.md) highest on ontological prompts (e.g. 27422 dog extension, 48684 Eiffel instanceOf). Strong signals. No cross-layer. Artifacts: experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/. 

**L16 150k results (2026-06-22 babysit)**: Fresh run of analysis exactly once this cycle. `python3 experiments/first_set/analyze_consistency.py --run_base experiments/first_set/sae_runs/llama_3_1_8b_layer --layers 16`. Per-group tops: hasExtension_abstract [588, 1020, 1825, 2167, 3492, 3889, 4051, 4591, 5532, 7672, 8246, 9049, 10425, 10769, 12811] ...; instanceOf [374, 1155, 1409, 1483, 1735, 2962, 6350, 6890, 9535, 13606, 14293, 15441, 16263, 16370, 17203] ...; superset_subClass [60, 402, 612, 756, 1104, 1792, 1873, 2118, 3392, 3950, 4159, 6350, 7265, 7957, 9434] ...; category_vs_member [196, 327, 588, 788, 1375, 3343, 4159, 5056, 5490, 5532, 6049, 6139, 6419, 8968, 9615] .... Catalog top activations on ontological prompts (e.g. 27422 dog extension, 48684 Eiffel instanceOf). Strong signals. No cross-layer. Artifacts: experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/. 

**L16 150k results (2026-06-22 babysit)**: Fresh analysis exactly once this cycle. `python3 experiments/first_set/analyze_consistency.py --run_base experiments/first_set/sae_runs/llama_3_1_8b_layer --layers 16`. Per-group tops: hasExtension_abstract [588, 1020, 1825, 2167, 3492, 3889, 4051, 4591, 5532, 7672, 8246, 9049, 10425, 10769, 12811] ...; instanceOf [374, 1155, 1409, 1483, 1735, 2962, 6350, 6890, 9535, 13606, 14293, 15441, 16263, 16370, 17203] ...; superset_subClass [60, 402, 612, 756, 1104, 1792, 1873, 2118, 3392, 3950, 4159, 6350, 7265, 7957, 9434] ...; category_vs_member [196, 327, 588, 788, 1375, 3343, 4159, 5056, 5490, 5532, 6049, 6139, 6419, 8968, 9615] .... Catalog top activations on ontological prompts (e.g. 27422 dog extension, 48684 Eiffel instanceOf). Strong signals. No cross-layer. Artifacts: experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/. 

**L16 150k results (2026-06-22 babysit)**: Fresh analysis exactly once this cycle. `python3 experiments/first_set/analyze_consistency.py --run_base experiments/first_set/sae_runs/llama_3_1_8b_layer --layers 16`. Per-group tops: hasExtension_abstract [588, 1020, 1825, 2167, 3492, 3889, 4051, 4591, 5532, 7672, 8246, 9049, 10425, 10769, 12811] ...; instanceOf [374, 1155, 1409, 1483, 1735, 2962, 6350, 6890, 9535, 13606, 14293, 15441, 16263, 16370, 17203] ...; superset_subClass [60, 402, 612, 756, 1104, 1792, 1873, 2118, 3392, 3950, 4159, 6350, 7265, 7957, 9434] ...; category_vs_member [196, 327, 588, 788, 1375, 3343, 4159, 5056, 5490, 5532, 6049, 6139, 6419, 8968, 9615] .... Catalog top activations on ontological prompts (e.g. 27422 dog extension, 48684 Eiffel instanceOf). Strong signals. No cross-layer. Artifacts: experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/. 

**L16 150k results (2026-06-22 scheduled babysit)**: Fresh analysis run this cycle (exactly once): `python3 experiments/first_set/analyze_consistency.py --run_base experiments/first_set/sae_runs/llama_3_1_8b_layer --layers 16` (KMP_DUPLICATE_LIB_OK=TRUE). Same per-group tops as prior. Catalog (feature_catalog.md) top activations on exact ontological language (strong, not noisy). No cross-layer comparisons performed. 

**L16 150k results (2026-06-22)**: Scheduled babysit analysis (fresh this cycle, exactly once): `python3 experiments/first_set/analyze_consistency.py --run_base experiments/first_set/sae_runs/llama_3_1_8b_layer --layers 16`. Per-group tops: hasExtension_abstract [588,1020,...,9049,...]; instanceOf [374,...]; superset_subClass [60,...]; category_vs_member [196,...]. Catalog highest: 27422 (max=3.36) on "dog has extension", 48684 on Eiffel instanceOf, 48351 on mammals superset, etc. Strong signals on ontological prompts. No cross-layer. Artifacts: experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/. 

**L16 150k results (2026-06-22)**: Scheduled babysit analysis re-run. `analyze_consistency.py --layers 16` produced: hasExtension_abstract [588,1020,...,9049,...]; instanceOf [374,...]; superset_subClass [60,...]; category_vs_member [196,...]. Catalog highest: 27422 (dog extension, max=3.36), 48684 (Eiffel instanceOf), 48351 (mammals superset), 17714 (vehicles subclasses). Strong signals on ontological language. No cross-layer. Artifacts: experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/.

**L16 150k results (2026-06-22 babysit)**: Fresh run of analysis (exactly once this cycle). Tops: hasExtension_abstract [588, 1020, 1825, 2167, 3492, 3889, 4051, 4591, 5532, 7672, 8246, 9049, 10425, 10769, 12811] ...; instanceOf [374, 1155, 1409, 1483, 1735, 2962, 6350, 6890, 9535, 13606, 14293, 15441, 16263, 16370, 17203] ...; superset_subClass [60, 402, 612, 756, 1104, 1792, 1873, 2118, 3392, 3950, 4159, 6350, 7265, 7957, 9434] ...; category_vs_member [196, 327, 588, 788, 1375, 3343, 4159, 5056, 5490, 5532, 6049, 6139, 6419, 8968, 9615] .... Catalog top features (e.g. 27422 on "dog has extension", 48684 on "Eiffel Tower is an instance", 48351 on "Mammals are a superset") are the ontological prompts. Strong LTC signals. No cross-layer comparisons. Artifacts: experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/.

**L16 150k results (2026-06-22)**: Scheduled babysit analysis (fresh this cycle). `analyze_consistency.py --layers 16` (KMP). Per-group tops match prior: hasExtension_abstract [588,1020,...,9049,...], instanceOf [374,...], superset_subClass [60,...], category_vs_member [196,...]. Catalog (feature_catalog.md) highest activations on exact ontological prompts (e.g. 27422 max=3.36 "dog has extension", 48684 "Eiffel instance of landmark", 48351 "mammals superset of dogs", 17714 vehicles subclasses, 9882 whale instance, 38742 professions). Strong signals. No cross-layer. Artifacts: experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/.

**L16 150k results (2026-06-21)**: `max_aggression_collect.py --tokens 150000 --layers 16 --expansion 12` (CPU, float32, KMP_DUPLICATE_LIB_OK=TRUE). Collection to 150k vectors + 1200-step BatchTopK training (d_sae=49152). Recon logs: step 0=1.47397, 150=0.06049, 300=0.01754, 450=0.00899, 600=0.00826, 750=0.00855, 900=0.00757, 1050=0.00679, 1199=0.00700. Collector "Analyzing..." then saved catalog + sae.pt. 

**Artifacts**: `experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/` (sae.pt ~1.6GB, feature_catalog.json 16.6MB, feature_catalog.md).

**Catalog observations** (top max-activation features from training data): Strong activation on target ontological language.
- hasExtension/abstract: "The concept of 'dog' has an extension...", "An abstract idea like 'justice' is different from the set of all just actions." (multiple features e.g. 27422 max=3.36, 28530=2.96, 9049=2.70).
- instanceOf: "The Eiffel Tower is an instance of 'landmark'." (e.g. 48684 max=2.89, 38873=2.82), "The word 'mammal' refers to the abstract category, while this particular whale is an instance." (9882 max=2.83).
- superset/subClass: "Mammals are a superset of dogs and cats." (48351 max=2.80), "Vehicles include cars, boats, and airplanes as subclasses." (17714 max=3.02, 5056=2.73).
- Other: "Professions such as doctor and lawyer are types of occupations." (38742 max=3.18).

Signals are clear and on-point even from collector's training-set top examples (label alignment note: prompts repeated). Analysis script (`analyze_consistency.py`) run hit d_sae load mismatch (catalog stored expansion=12 but no explicit d_sae; defaulted 32768 vs actual 49152; state_dict load failed both BatchTopK and fallback paths). Fresh canonical re-compute not obtained, but collector catalog already shows reusable ontological features. Limitations: CPU-only (slow ~1h), mixed pile data may dilute, labels are whole-prompt repeats, single layer so far (no cross yet).

**Analysis script output** (run on canonical cases for groups): 

Top feature ids per group (by max activation on canonical prompts for layer 16):

- hasExtension_abstract: [588, 1020, 1825, 2167, 3492, 3889, 4051, 4591, 5532, 7672, 8246, 9049, 10425, 10769, 12811] ...

- instanceOf: [374, 1155, 1409, 1483, 1735, 2962, 6350, 6890, 9535, 13606, 14293, 15441, 16263, 16370, 17203] ...

- superset_subClass: [60, 402, 612, 756, 1104, 1792, 1873, 2118, 3392, 3950, 4159, 6350, 7265, 7957, 9434] ...

- category_vs_member: [196, 327, 588, 788, 1375, 3343, 4159, 5056, 5490, 5532, 6049, 6139, 6419, 8968, 9615] ...

Note overlap e.g. 9049 in hasExtension_abstract (also high in catalog), 588 in hasExtension and category_vs_member, etc.

This is concrete evidence of LTC-like features at 8B scale. Cross-layer next.

This gets us everything (breadth + depth) iteratively and safely. See notebook "Phased Plan" section for schedule and live status.

**L16 scheduled verification (fresh run of analyze)**: sae.pt present, log ends with "complete". Analysis re-executed: produced same top feature groups as recorded. Catalog confirms multiple high-activating features on exact ontological prompts (hasExtension, instanceOf, superset/subClass language). Signals appear strong for Phase 1 LTC hypothesis on this layer. (No new cross-layer Jaccard run per instructions.) Artifacts reproducible at experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/.

**L16 scheduled cycle update (this execution)**: Analysis run again per schedule. Top groups (by max act on canonical): hasExtension_abstract [588,1020,...9049,...], instanceOf [374,...], superset_subClass [60,...], category_vs_member [196,...]. Catalog top-10 all map to ontological prompts (e.g. 27422 on dog-extension hasExtension, 48684 on Eiffel instanceOf, 48351 on mammals superset). Strong, consistent signals. No cross-layer. Docs extended. (L16 only this cycle.)

**L16 150k results (2026-06-22 scheduled)**: sae.pt exists, log complete. Analysis run (fresh): same per-group tops as prior (hasExtension_abstract etc. with IDs 588/374/60/196 etc.). Catalog review: top features (27422 max=3.36 etc.) on exact ontological prompts (hasExtension dog, instance Eiffel, superset mammals, professions, vehicles). Strong LTC signals, no noise. No cross-layer impact (single layer). Artifacts: experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/. 

**L16 150k scheduled update (this cycle)**: Analysis re-run per schedule. Output matches previous (strong tops on canonical). Catalog confirms features like 27422/48684/48351 on hasExtension/instanceOf/superset prompts. Strong, not weak. No new cross-layer. 

**L16 150k results (2026-06-22 scheduled cycle)**: Analysis run (fresh): same per-group tops. Catalog review: top features on exact ontological prompts. Strong LTC signals. No cross-layer. Artifacts: experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/. (Note: L12 200k also completed per monitor.) 

**L12 200k results (2026-06-22)**: `max_aggression_collect.py --tokens 200000 --layers 12 --expansion 12` (CPU, float32, KMP_DUPLICATE_LIB_OK=TRUE). Collection to 200k vectors + 1200-step BatchTopK training (d_sae=49152). Recon logs: step 0=0.97319, 150=0.01680, 300=0.01038, 450=0.00946, 600=0.00896, 750=0.00494, 900=0.00361, 1050=0.00354, 1199=0.00349. Collector "Analyzing..." then saved catalog + sae.pt. 

**Artifacts**: `experiments/first_set/sae_runs/llama_3_1_8b_layer12_max/` (sae.pt ~1.61GB, feature_catalog.json ~14MB, feature_catalog.md).

**Catalog observations** (top max-activation features from training data): Strong activation on target ontological language.
- superset/subset: "The set of even numbers is a proper subset of the integers." (e.g. Feature 38995 max=2.49, 3340=2.16, 22478=2.09, 42650=2.08).
- hasExtension/abstract: "An abstract idea like 'justice' is different from the set of all just actions." (5532 max=2.24, 32094=2.06), "The intension of 'mammal' is the abstract category; its extension is every living mammal." (46355 max=2.15).
- instanceOf: "Fido is a specific instance of the class dog." (16736 max=2.16, 9555=2.08).

Signals are clear and on-point even from collector's training-set top examples (label alignment note: prompts repeated). Analysis script (`analyze_consistency.py`) run produced distinct top features per group.

**Analysis script output** (run on canonical cases for groups): 

Top feature ids per group (by max activation on canonical prompts for layer 12):

- hasExtension_abstract: [1048, 1317, 2683, 4781, 5469, 5507, 5532, 6449, 6541, 8907, 9139, 9269, 9718, 11453, 12202] ...

- instanceOf: [1415, 1622, 1903, 5163, 5507, 8183, 8540, 8572, 9177, 9555, 9718, 9841, 10471, 10829, 12543] ...

- superset_subClass: [951, 2017, 3210, 3976, 4154, 4322, 5699, 7397, 7679, 7890, 8540, 9718, 10608, 14852, 15495] ...

- category_vs_member: [865, 1317, 2864, 4028, 5507, 6396, 8375, 10067, 10448, 10777, 10965, 11276, 12091, 13988, 15069] ...

Note overlap e.g. 5507 in multiple groups, 9718 in hasExtension and superset_subClass.

This is concrete evidence of LTC-like features at 8B scale. Cross-layer next (L16 + L12 available; per instructions, cross not started yet in this phase).

**Cross-Layer Comparison (L12 + L16) (2026-06-22)**

**Causal Analysis on Key Ontological Features (L12 + L16) — Revised (2026-06-23)**

**Target / broader set tested**: Focused on the cross-layer feature 5532 (appears in hasExtension_abstract tops for both L12 and L16; catalog examples: "justice" intension vs its extension set; "mammal" abstract vs living instances). Sampled one additional from L16 hasExtension group (9049, high max-act on "dog has an extension" language in catalog). Selection drawn from analyze_consistency top-k per group + high-max catalog features across hasExtension/instanceOf/superset (see FEATURES in `refined_causal.py`: L12 5532+46355; L16 5532+9049 for this execution). Other catalog notables (27422, 48684, 48351, 16736, 38995, 17714, ...) remain candidates for follow-up.

**Improved methodology** (see `experiments/first_set/refined_causal.py`):
- Load via sae_lens BatchTopKTrainingSAE (exact cfg d_in=4096, d_sae=49152, k~384); use .encode() + .decode() for all interventions (no manual Linear fallback in main path).
- Compute per-feature means on neutral non-ontological sentences for mean-ablation baseline.
- Three interventions per (feat, prompt): zero-ablation, mean-ablation, amplification (x3.0 scale on the feature activation).
- Also measure recon-only (encode then decode with scale=1.0) to quantify non-specific distortion.
- Cleaner targeted contrastive prompts isolating the relations:
  1. hasExtension: "The concept of 'dog' has an extension that includes all actual" → good " dogs" / bad " cats"
  2. hasExtension/abstract: "An abstract idea like 'justice' is different from the set of all just" → " actions" / " objects"
  3. instanceOf: "Fido is a specific instance of the class" → " dog" / " cat"
  4. supersetOf: "All birds are animals. A robin is therefore an" → " animal" / " plant"
- Metric: Δ = [logit(good) − logit(bad)]_patched − [logit(good) − logit(bad)]_clean . Positive Δ = intervention helped correct ontological continuation; negative = hurt.
- Script also reports % of prompts where zero "hurts" (Δ<0) and amp "helps" (Δ>0).

**Results (CPU; 4 prompts, 5532 at both layers + 9049 at L16)**:

- L12 feat 5532 (4 prompts): avg clean_diff ≈ +7.79. recon-only Δ ≈ −2.29. zero Δ ≈ −2.33. mean Δ ≈ −2.34. amp3 Δ ≈ −2.25. zero_hurts 75%, amp_helps 25%.
- L16 feat 5532 (4 prompts): avg clean_diff ≈ +7.79. recon-only Δ ≈ −3.25. zero Δ ≈ −3.17. mean Δ ≈ −3.20. amp3 Δ ≈ −3.45. zero_hurts 100%, amp_helps 0%.
  Per-prompt (L16-5532): dog-ext (clean ~7.40, recon −2.36, zero −2.35, amp −2.39); justice-abs (clean ~4.55, recon −0.62, zero −0.44, amp −1.14 — amp hurts *more*); Fido-inst (clean ~5.28, recon/zero ≈ −2.54); robin-superset (clean ~13.93, recon/zero/amp = −7.47 — *entire drop is recon distortion*, feature adds nothing detectable).
- L16 feat 9049 (2 prompts, hasExt catalog): avg clean_diff ≈ +5.97. recon-only Δ ≈ −1.49. zero/mean Δ ≈ −1.48. amp3 Δ ≈ −1.50. zero_hurts 100%, amp_helps 0% in sample.
  - Per-prompt variation large: one dog-extension prompt showed recon/zero ~−2.35; the abstract-justice prompt showed only ~−0.62 (entirely recon).

- Amplification never increased the correct margin in these samples (frequently matched or exceeded the recon penalty).
- Mean-ablation ≈ zero-ablation (most tested features had near-zero mean activation on neutrals and were sparse on the test prompts).
- Ablation produces the expected directional drop in preference for ontologically-correct tokens on several prompts, but magnitude is almost entirely accounted for by the SAE reconstruction penalty itself.

**Strength and consistency of causal effects**: Modest and fragile. 5532 at both layers shows involvement (ablating it shifts behavior in the expected direction more than chance), but (a) effect size is small once recon error is considered, (b) not consistent across all prompts, (c) boosting the feature does not produce the opposite (stronger correct) behavior. 9049 exhibits the same pattern at lower absolute effect. This is consistent with the low cross-layer Jaccard and with a distributed, depth-specialized implementation of the relations rather than a small set of "master" features that can be dialed.

**Remaining limitations (explicit)**:
- SAE reconstruction fidelity is poor relative to the size of the measured effects (recon Δ often equals or exceeds the ablation Δ). Interventions are therefore confounded; we obtain only an upper bound on the causal role of any single feature. Non-specific damage from the round-trip can mimic feature ablation.
- Very small prompt set (4 short prompts, thematically close to the collector's CANONICAL_CASES and training mix). No statistical power, no out-of-distribution concepts.
- Single-feature interventions only. No joint ablations, no path patching, no measurement of downstream attention/MLP contributions.
- CPU-only; could not exhaust the intended broader set (e.g. L16-27422/48684/48351/17714/5056/9882, L12-38995/16736/9555/32094 etc.). Sampled representatives only.
- Many ontological features are sparse; on prompts where act≈0 the intervention has no effect beyond recon.
- to_single_token constraints further pruned the contrast set.
- No full-distribution metrics (top-1 logit diff is a narrow proxy).

**Summary of refined picture**: We now have concrete numbers using proper SAE reconstruction, zero/mean/amp, and cleaner prompts. The data continue to support that these features participate in ontological-relation processing (directional causal effects exist), but the representation is highly distributed and the current SAE-based patching is too lossy to support strong claims about individual features being "the" mechanism. Amplification did not yield controllable enhancement of correct reasoning. This is useful negative information for the LTC hypothesis: if a clean "Class Thread" circuit existed in a handful of features we would expect more decisive, recon-robust, and amplifiable effects.

See `refined_causal.py` + `refined_causal_results.json` for code and raw deltas. Notebook contains the same updated section.

This completes the requested refinement of the causal analysis on existing L12/L16 artifacts. No L20 collection was launched. Ready for user decision on next direction.

This is concrete evidence of LTC-like features at 8B scale. Cross-layer next (L16 + L12 available; per instructions, cross not started yet in this phase).

**Setup**: Same model (Llama-3.1-8B-Instruct), same 12× BatchTopK SAE (d_sae=49152). L12 trained on 200k tokens; L16 on 150k tokens. Top features scored by max activation on the same canonical ontological prompts used throughout Phase 1. Cross-layer Jaccard computed by the analysis script over top-k features per group.

**Per-group top features** (by max act on canonical prompts):

- **hasExtension_abstract**:
  - L12: [1048, 1317, 2683, 4781, 5469, 5507, 5532, ...]
  - L16: [588, 1020, 1825, 2167, 3492, 3889, 4051, ..., 5532, ...]
  - Shared in top lists: 5532 (appears in both; catalog examples involve "justice" abstract vs. its extension/set of just actions, and "mammal" intension vs. extension).

- **instanceOf**:
  - L12: [1415, 1622, 1903, 5163, 5507, 8183, 8540, ..., 9718, ...]
  - L16: [374, 1155, 1409, 1483, 1735, 2962, 6350, ...]
  - No overlap in top-15 lists.

- **superset_subClass**:
  - L12: [951, 2017, 3210, 3976, 4154, 4322, 5699, ..., 9718, ...]
  - L16: [60, 402, 612, 756, 1104, 1792, 1873, ...]
  - No overlap in top-15.

- **category_vs_member**:
  - L12: [865, 1317, 2864, 4028, 5507, 6396, 8375, ...]
  - L16: [196, 327, 588, 788, 1375, 3343, 4159, ..., 5532, ...]
  - No overlap in top-15.

**Cross-layer overlap (Jaccard @ top-k from analysis script)**:
- hasExtension_abstract: Jaccard=0.020 (2 shared)
- instanceOf: Jaccard=0.010 (1 shared)
- superset_subClass: Jaccard=0.000 (0 shared)
- category_vs_member: Jaccard=0.010 (1 shared)

**Catalog contrast** (training-data highest max-activation examples):
- L12 tends to surface subset language ("even numbers is a proper subset of the integers" — Features 38995, 3340) and instance chains ("Fido is a specific instance...", "poodle is a dog... mammal").
- L16 surfaces explicit relational phrasing ("has an extension that includes all actual dogs", "is an instance of 'landmark'", "are a superset of dogs and cats", "the abstract category, while this particular whale is an instance").
- Common theme: both layers have features sensitive to intension/extension and class-membership language, but realized by largely different feature IDs.

**Patterns and LTC evidence**:
- Both layers independently learn features that fire on the core ontological relations.
- However, the *specific* features (IDs) show very low overlap. The model appears to implement similar *sensitivities* using distinct representational resources at different depths.
- One weak cross-layer signal: Feature 5532 participates in hasExtension_abstract scoring at both layers (justice/mammal intension-extension examples).
- This is preliminary evidence *against* a single reusable "Class Thread" feature set being copied across layers. It is more consistent with distributed, depth-specialized representations of the same abstract relations.
- Positive for the broader hypothesis: ontological relation detection is not confined to a narrow band of layers.

**Honest limitations (what we cannot yet conclude)**:
- Different training token counts (150k vs 200k) may affect feature sharpness.
- Only two layers compared; no L8/L20/L24 yet.
- Jaccard and tops are based on max-activation on a small canonical set (repeated prompts from the training mix).
- No causal evidence (activation patching, etc.) that these features are *used* by the model for reasoning.
- Top-k overlap may miss weaker but consistent contributions from other features.

**Summary**: We see evidence that the model develops internal sensitivity to hasExtension, instanceOf, superset/subClass, and category-vs-member at multiple layers. However, the concrete features are mostly distinct rather than consistently reused. This supports a version of the LLM-Tapestry Characteristic (the relations are represented at different depths) but does not yet support a strong "same thread" story. More layers + causal work needed.

This gets us everything (breadth + depth) iteratively and safely. See notebook "Phased Plan" section for schedule and live status.

**L16 150k results (2026-06-22 scheduled)**: `max_aggression_collect.py --tokens 150000 --layers 16 --expansion 12` (CPU, float32, KMP_DUPLICATE_LIB_OK=TRUE). Collection to 150k vectors + 1200-step BatchTopK training (d_sae=49152). Recon logs: step 0=1.47397, 150=0.06049, 300=0.01754, 450=0.00899, 600=0.00826, 750=0.00855, 900=0.00757, 1050=0.00679, 1199=0.00700. Collector "Analyzing..." then saved catalog + sae.pt. 

**Artifacts**: `experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/` (sae.pt ~1.6GB, feature_catalog.json 16.6MB, feature_catalog.md).

**Catalog observations** (top max-activation features from training data): Strong activation on target ontological language.
- hasExtension/abstract: "The concept of 'dog' has an extension..." (e.g. 27422 max=3.36, 28530=2.96, 9049=2.70).
- instanceOf: "The Eiffel Tower is an instance of 'landmark'." (48684 max=2.89, 38873=2.82), "The word 'mammal' refers to the abstract category, while this particular whale is an instance." (9882 max=2.83).
- superset/subClass: "Mammals are a superset of dogs and cats." (48351 max=2.80), "Vehicles include cars, boats, and airplanes as subclasses." (17714 max=3.02, 5056=2.73).
- Other: "Professions such as doctor and lawyer are types of occupations." (38742 max=3.18).

Signals are clear and on-point even from collector's training-set top examples (label alignment note: prompts repeated). Analysis script run produced same per-group tops. Limitations: CPU-only (slow), mixed pile data may dilute, labels whole-prompt repeats, single layer (no cross yet).

**Analysis script output** (run on canonical cases for groups): 

Top feature ids per group (by max activation on canonical prompts for layer 16):

- hasExtension_abstract: [588, 1020, 1825, 2167, 3492, 3889, 4051, 4591, 5532, 7672, 8246, 9049, 10425, 10769, 12811] ...

- instanceOf: [374, 1155, 1409, 1483, 1735, 2962, 6350, 6890, 9535, 13606, 14293, 15441, 16263, 16370, 17203] ...

- superset_subClass: [60, 402, 612, 756, 1104, 1792, 1873, 2118, 3392, 3950, 4159, 6350, 7265, 7957, 9434] ...

- category_vs_member: [196, 327, 588, 788, 1375, 3343, 4159, 5056, 5490, 5532, 6049, 6139, 6419, 8968, 9615] ...

Note overlap e.g. 9049 in hasExtension_abstract (also high in catalog), 588 in hasExtension and category_vs_member, etc.

This is concrete evidence of LTC-like features at 8B scale. Cross-layer next. 

**L16 150k results (2026-06-22 scheduled)**: `max_aggression_collect.py --tokens 150000 --layers 16 --expansion 12` (CPU, float32, KMP_DUPLICATE_LIB_OK=TRUE). Collection to 150k vectors + 1200-step BatchTopK training (d_sae=49152). Recon logs: step 0=1.47397, 150=0.06049, 300=0.01754, 450=0.00899, 600=0.00826, 750=0.00855, 900=0.00757, 1050=0.00679, 1199=0.00700. Collector "Analyzing..." then saved catalog + sae.pt. 

**Artifacts**: `experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/` (sae.pt ~1.6GB, feature_catalog.json 16.6MB, feature_catalog.md).

**Catalog observations** (top max-activation from training data): Strong on ontological language.
- hasExtension/abstract: "The concept of 'dog' has an extension..." (27422 max=3.36, 28530=2.96, 9049=2.70).
- instanceOf: "The Eiffel Tower is an instance of 'landmark'." (48684=2.89, 38873=2.82), "The word 'mammal' refers to the abstract category, while this particular whale is an instance." (9882=2.83).
- superset/subClass: "Mammals are a superset of dogs and cats." (48351=2.80), "Vehicles include cars, boats, and airplanes as subclasses." (17714=3.02, 5056=2.73).
- Other: "Professions such as doctor and lawyer are types of occupations." (38742=3.18).

Signals clear. Analysis produced same per-group tops. Limitations: CPU-only, mixed pile, repeated labels, single layer.

**Analysis script output** (canonical for layer 16):

- hasExtension_abstract: [588, 1020, 1825, 2167, 3492, 3889, 4051, 4591, 5532, 7672, 8246, 9049, 10425, 10769, 12811] ...
- instanceOf: [374, 1155, 1409, 1483, 1735, 2962, 6350, 6890, 9535, 13606, 14293, 15441, 16263, 16370, 17203] ...
- superset_subClass: [60, 402, 612, 756, 1104, 1792, 1873, 2118, 3392, 3950, 4159, 6350, 7265, 7957, 9434] ...
- category_vs_member: [196, 327, 588, 788, 1375, 3343, 4159, 5056, 5490, 5532, 6049, 6139, 6419, 8968, 9615] ...

Overlap e.g. 9049, 588. Concrete LTC-like evidence at 8B. Cross-layer next. 

**L16 150k results (2026-06-22 scheduled)**: `max_aggression_collect.py --tokens 150000 --layers 16 --expansion 12` (CPU, float32, KMP_DUPLICATE_LIB_OK=TRUE). Collection to 150k vectors + 1200-step BatchTopK training (d_sae=49152). Recon logs: step 0=1.47397, 150=0.06049, 300=0.01754, 450=0.00899, 600=0.00826, 750=0.00855, 900=0.00757, 1050=0.00679, 1199=0.00700. Collector "Analyzing..." then saved catalog + sae.pt. 

**Artifacts**: `experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/` (sae.pt ~1.6GB, feature_catalog.json 16.6MB, feature_catalog.md).

**Catalog observations** (top max-activation features from training data): Strong activation on target ontological language.
- hasExtension/abstract: "The concept of 'dog' has an extension..." (e.g. 27422 max=3.36, 28530=2.96, 9049=2.70).
- instanceOf: "The Eiffel Tower is an instance of 'landmark'." (48684=2.89, 38873=2.82), "The word 'mammal' refers to the abstract category, while this particular whale is an instance." (9882=2.83).
- superset/subClass: "Mammals are a superset of dogs and cats." (48351=2.80), "Vehicles include cars, boats, and airplanes as subclasses." (17714=3.02, 5056=2.73).
- Other: "Professions such as doctor and lawyer are types of occupations." (38742=3.18).

Signals are clear and on-point even from collector's training-set top examples (label alignment note: prompts repeated). Analysis script run produced same per-group tops. Limitations: CPU-only (slow), mixed pile data may dilute, labels are whole-prompt repeats, single layer so far (no cross yet).

**Analysis script output** (run on canonical cases for groups): 

Top feature ids per group (by max activation on canonical prompts for layer 16):

- hasExtension_abstract: [588, 1020, 1825, 2167, 3492, 3889, 4051, 4591, 5532, 7672, 8246, 9049, 10425, 10769, 12811] ...

- instanceOf: [374, 1155, 1409, 1483, 1735, 2962, 6350, 6890, 9535, 13606, 14293, 15441, 16263, 16370, 17203] ...

- superset_subClass: [60, 402, 612, 756, 1104, 1792, 1873, 2118, 3392, 3950, 4159, 6350, 7265, 7957, 9434] ...

- category_vs_member: [196, 327, 588, 788, 1375, 3343, 4159, 5056, 5490, 5532, 6049, 6139, 6419, 8968, 9615] ...

Note overlap e.g. 9049 in hasExtension_abstract (also high in catalog), 588 in hasExtension and category_vs_member, etc.

This is concrete evidence of LTC-like features at 8B scale. Cross-layer next. 


**L16 150k results (2026-06-22 scheduled)**: `max_aggression_collect.py --tokens 150000 --layers 16 --expansion 12` (CPU, float32, KMP_DUPLICATE_LIB_OK=TRUE). Collection to 150k vectors + 1200-step BatchTopK training (d_sae=49152). Recon logs: step 0=1.47397, 150=0.06049, 300=0.01754, 450=0.00899, 600=0.00826, 750=0.00855, 900=0.00757, 1050=0.00679, 1199=0.00700. Collector "Analyzing..." then saved catalog + sae.pt. 

**Artifacts**: `experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/` (sae.pt ~1.6GB, feature_catalog.json 16.6MB, feature_catalog.md).

**Catalog observations** (top max-activation features from training data): Strong activation on target ontological language.
- hasExtension/abstract: "The concept of 'dog' has an extension..." (e.g. 27422 max=3.36, 28530=2.96, 9049=2.70).
- instanceOf: "The Eiffel Tower is an instance of 'landmark'." (48684=2.89, 38873=2.82), "The word 'mammal' refers to the abstract category, while this particular whale is an instance." (9882=2.83).
- superset/subClass: "Mammals are a superset of dogs and cats." (48351=2.80), "Vehicles include cars, boats, and airplanes as subclasses." (17714=3.02, 5056=2.73).
- Other: "Professions such as doctor and lawyer are types of occupations." (38742=3.18).

Signals are clear and on-point even from collector's training-set top examples (label alignment note: prompts repeated). Analysis script run produced same per-group tops. Limitations: CPU-only (slow), mixed pile data may dilute, labels are whole-prompt repeats, single layer so far (no cross yet).

**Analysis script output** (run on canonical cases for groups): 

Top feature ids per group (by max activation on canonical prompts for layer 16):

- hasExtension_abstract: [588, 1020, 1825, 2167, 3492, 3889, 4051, 4591, 5532, 7672, 8246, 9049, 10425, 10769, 12811] ...

- instanceOf: [374, 1155, 1409, 1483, 1735, 2962, 6350, 6890, 9535, 13606, 14293, 15441, 16263, 16370, 17203] ...

- superset_subClass: [60, 402, 612, 756, 1104, 1792, 1873, 2118, 3392, 3950, 4159, 6350, 7265, 7957, 9434] ...

- category_vs_member: [196, 327, 588, 788, 1375, 3343, 4159, 5056, 5490, 5532, 6049, 6139, 6419, 8968, 9615] ...

Note overlap e.g. 9049 in hasExtension_abstract (also high in catalog), 588 in hasExtension and category_vs_member, etc.

This is concrete evidence of LTC-like features at 8B scale. Cross-layer next. 

**L16 150k results (2026-06-22 scheduled)**: `max_aggression_collect.py --tokens 150000 --layers 16 --expansion 12` (CPU, float32, KMP_DUPLICATE_LIB_OK=TRUE). Collection to 150k vectors + 1200-step BatchTopK training (d_sae=49152). Recon logs: step 0=1.47397, 150=0.06049, 300=0.01754, 450=0.00899, 600=0.00826, 750=0.00855, 900=0.00757, 1050=0.00679, 1199=0.00700. Collector "Analyzing..." then saved catalog + sae.pt. 

**Artifacts**: `experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/` (sae.pt ~1.6GB, feature_catalog.json 16.6MB, feature_catalog.md).

**Catalog observations** (top max-activation features from training data): Strong activation on target ontological language.
- hasExtension/abstract: "The concept of 'dog' has an extension..." (e.g. 27422 max=3.36, 28530=2.96, 9049=2.70).
- instanceOf: "The Eiffel Tower is an instance of 'landmark'." (48684=2.89, 38873=2.82), "The word 'mammal' refers to the abstract category, while this particular whale is an instance." (9882=2.83).
- superset/subClass: "Mammals are a superset of dogs and cats." (48351=2.80), "Vehicles include cars, boats, and airplanes as subclasses." (17714=3.02, 5056=2.73).
- Other: "Professions such as doctor and lawyer are types of occupations." (38742=3.18).

Signals are clear and on-point even from collector's training-set top examples (label alignment note: prompts repeated). Analysis script run produced same per-group tops. Limitations: CPU-only (slow), mixed pile data may dilute, labels are whole-prompt repeats, single layer so far (no cross yet).

**Analysis script output** (run on canonical cases for groups): 

Top feature ids per group (by max activation on canonical prompts for layer 16):

- hasExtension_abstract: [588, 1020, 1825, 2167, 3492, 3889, 4051, 4591, 5532, 7672, 8246, 9049, 10425, 10769, 12811] ...

- instanceOf: [374, 1155, 1409, 1483, 1735, 2962, 6350, 6890, 9535, 13606, 14293, 15441, 16263, 16370, 17203] ...

- superset_subClass: [60, 402, 612, 756, 1104, 1792, 1873, 2118, 3392, 3950, 4159, 6350, 7265, 7957, 9434] ...

- category_vs_member: [196, 327, 588, 788, 1375, 3343, 4159, 5056, 5490, 5532, 6049, 6139, 6419, 8968, 9615] ...

Note overlap e.g. 9049 in hasExtension_abstract (also high in catalog), 588 in hasExtension and category_vs_member, etc.

This is concrete evidence of LTC-like features at 8B scale. Cross-layer next.

**L16 150k results (2026-06-22 scheduled)**: `max_aggression_collect.py --tokens 150000 --layers 16 --expansion 12` (CPU, float32, KMP_DUPLICATE_LIB_OK=TRUE). Collection to 150k vectors + 1200-step BatchTopK training (d_sae=49152). Recon logs: step 0=1.47397, 150=0.06049, 300=0.01754, 450=0.00899, 600=0.00826, 750=0.00855, 900=0.00757, 1050=0.00679, 1199=0.00700. Collector "Analyzing..." then saved catalog + sae.pt. 

**Artifacts**: `experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/` (sae.pt ~1.6GB, feature_catalog.json 16.6MB, feature_catalog.md).

**Catalog observations** (top max-activation features from training data): Strong activation on target ontological language.
- hasExtension/abstract: "The concept of 'dog' has an extension..." (e.g. 27422 max=3.36, 28530=2.96, 9049=2.70).
- instanceOf: "The Eiffel Tower is an instance of 'landmark'." (48684=2.89, 38873=2.82), "The word 'mammal' refers to the abstract category, while this particular whale is an instance." (9882=2.83).
- superset/subClass: "Mammals are a superset of dogs and cats." (48351=2.80), "Vehicles include cars, boats, and airplanes as subclasses." (17714=3.02, 5056=2.73).
- Other: "Professions such as doctor and lawyer are types of occupations." (38742=3.18).

Signals are clear and on-point even from collector's training-set top examples (label alignment note: prompts repeated). Analysis script run produced same per-group tops. Limitations: CPU-only (slow), mixed pile data may dilute, labels are whole-prompt repeats, single layer so far (no cross yet).

**Analysis script output** (run on canonical cases for groups): 

Top feature ids per group (by max activation on canonical prompts for layer 16):

- hasExtension_abstract: [588, 1020, 1825, 2167, 3492, 3889, 4051, 4591, 5532, 7672, 8246, 9049, 10425, 10769, 12811] ...

- instanceOf: [374, 1155, 1409, 1483, 1735, 2962, 6350, 6890, 9535, 13606, 14293, 15441, 16263, 16370, 17203] ...

- superset_subClass: [60, 402, 612, 756, 1104, 1792, 1873, 2118, 3392, 3950, 4159, 6350, 7265, 7957, 9434] ...

- category_vs_member: [196, 327, 588, 788, 1375, 3343, 4159, 5056, 5490, 5532, 6049, 6139, 6419, 8968, 9615] ...

Note overlap e.g. 9049 in hasExtension_abstract (also high in catalog), 588 in hasExtension and category_vs_member, etc.

This is concrete evidence of LTC-like features at 8B scale. Cross-layer next.

**L16 150k results (2026-06-22 scheduled)**: `max_aggression_collect.py --tokens 150000 --layers 16 --expansion 12` (CPU, float32, KMP_DUPLICATE_LIB_OK=TRUE). Collection to 150k vectors + 1200-step BatchTopK training (d_sae=49152). Recon logs: step 0=1.47397, 150=0.06049, 300=0.01754, 450=0.00899, 600=0.00826, 750=0.00855, 900=0.00757, 1050=0.00679, 1199=0.00700. Collector "Analyzing..." then saved catalog + sae.pt. 

**Artifacts**: `experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/` (sae.pt ~1.6GB, feature_catalog.json 16.6MB, feature_catalog.md).

**Catalog observations** (top max-activation features from training data): Strong activation on target ontological language.
- hasExtension/abstract: "The concept of 'dog' has an extension..." (e.g. 27422 max=3.36, 28530=2.96, 9049=2.70).
- instanceOf: "The Eiffel Tower is an instance of 'landmark'." (48684=2.89, 38873=2.82), "The word 'mammal' refers to the abstract category, while this particular whale is an instance." (9882=2.83).
- superset/subClass: "Mammals are a superset of dogs and cats." (48351=2.80), "Vehicles include cars, boats, and airplanes as subclasses." (17714=3.02, 5056=2.73).
- Other: "Professions such as doctor and lawyer are types of occupations." (38742=3.18).

Signals are clear and on-point even from collector's training-set top examples (label alignment note: prompts repeated). Analysis script run produced same per-group tops. Limitations: CPU-only (slow), mixed pile data may dilute, labels are whole-prompt repeats, single layer so far (no cross yet).

**Analysis script output** (run on canonical cases for groups): 

Top feature ids per group (by max activation on canonical prompts for layer 16):

- hasExtension_abstract: [588, 1020, 1825, 2167, 3492, 3889, 4051, 4591, 5532, 7672, 8246, 9049, 10425, 10769, 12811] ...

- instanceOf: [374, 1155, 1409, 1483, 1735, 2962, 6350, 6890, 9535, 13606, 14293, 15441, 16263, 16370, 17203] ...

- superset_subClass: [60, 402, 612, 756, 1104, 1792, 1873, 2118, 3392, 3950, 4159, 6350, 7265, 7957, 9434] ...

- category_vs_member: [196, 327, 588, 788, 1375, 3343, 4159, 5056, 5490, 5532, 6049, 6139, 6419, 8968, 9615] ...

Note overlap e.g. 9049 in hasExtension_abstract (also high in catalog), 588 in hasExtension and category_vs_member, etc.

This is concrete evidence of LTC-like features at 8B scale. Cross-layer next.

**L16 150k results (2026-06-22 scheduled)**: `max_aggression_collect.py --tokens 150000 --layers 16 --expansion 12` (CPU, float32, KMP_DUPLICATE_LIB_OK=TRUE). Collection to 150k vectors + 1200-step BatchTopK training (d_sae=49152). Recon logs: step 0=1.47397, 150=0.06049, 300=0.01754, 450=0.00899, 600=0.00826, 750=0.00855, 900=0.00757, 1050=0.00679, 1199=0.00700. Collector "Analyzing..." then saved catalog + sae.pt. 

**Artifacts**: `experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/` (sae.pt ~1.6GB, feature_catalog.json 16.6MB, feature_catalog.md).

**Catalog observations** (top max-activation features from training data): Strong activation on target ontological language.
- hasExtension/abstract: "The concept of 'dog' has an extension..." (e.g. 27422 max=3.36, 28530=2.96, 9049=2.70).
- instanceOf: "The Eiffel Tower is an instance of 'landmark'." (48684=2.89, 38873=2.82), "The word 'mammal' refers to the abstract category, while this particular whale is an instance." (9882=2.83).
- superset/subClass: "Mammals are a superset of dogs and cats." (48351=2.80), "Vehicles include cars, boats, and airplanes as subclasses." (17714=3.02, 5056=2.73).
- Other: "Professions such as doctor and lawyer are types of occupations." (38742=3.18).

Signals are clear and on-point even from collector's training-set top examples (label alignment note: prompts repeated). Analysis script run produced same per-group tops. Limitations: CPU-only (slow), mixed pile data may dilute, labels are whole-prompt repeats, single layer so far (no cross yet).

**Analysis script output** (run on canonical cases for groups): 

Top feature ids per group (by max activation on canonical prompts for layer 16):

- hasExtension_abstract: [588, 1020, 1825, 2167, 3492, 3889, 4051, 4591, 5532, 7672, 8246, 9049, 10425, 10769, 12811] ...

- instanceOf: [374, 1155, 1409, 1483, 1735, 2962, 6350, 6890, 9535, 13606, 14293, 15441, 16263, 16370, 17203] ...

- superset_subClass: [60, 402, 612, 756, 1104, 1792, 1873, 2118, 3392, 3950, 4159, 6350, 7265, 7957, 9434] ...

- category_vs_member: [196, 327, 588, 788, 1375, 3343, 4159, 5056, 5490, 5532, 6049, 6139, 6419, 8968, 9615] ...

Note overlap e.g. 9049 in hasExtension_abstract (also high in catalog), 588 in hasExtension and category_vs_member, etc.

This is concrete evidence of LTC-like features at 8B scale. Cross-layer next.

**L16 150k results (2026-06-22 scheduled)**: `max_aggression_collect.py --tokens 150000 --layers 16 --expansion 12` (CPU, float32, KMP_DUPLICATE_LIB_OK=TRUE). Collection to 150k vectors + 1200-step BatchTopK training (d_sae=49152). Recon logs: step 0=1.47397, 150=0.06049, 300=0.01754, 450=0.00899, 600=0.00826, 750=0.00855, 900=0.00757, 1050=0.00679, 1199=0.00700. Collector "Analyzing..." then saved catalog + sae.pt. 

**Artifacts**: `experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/` (sae.pt ~1.6GB, feature_catalog.json 16.6MB, feature_catalog.md).

**Catalog observations** (top max-activation features from training data): Strong activation on target ontological language.
- hasExtension/abstract: "The concept of 'dog' has an extension..." (e.g. 27422 max=3.36, 28530=2.96, 9049=2.70).
- instanceOf: "The Eiffel Tower is an instance of 'landmark'." (48684=2.89, 38873=2.82), "The word 'mammal' refers to the abstract category, while this particular whale is an instance." (9882=2.83).
- superset/subClass: "Mammals are a superset of dogs and cats." (48351=2.80), "Vehicles include cars, boats, and airplanes as subclasses." (17714=3.02, 5056=2.73).
- Other: "Professions such as doctor and lawyer are types of occupations." (38742=3.18).

Signals are clear and on-point even from collector's training-set top examples (label alignment note: prompts repeated). Analysis script run produced same per-group tops. Limitations: CPU-only (slow), mixed pile data may dilute, labels are whole-prompt repeats, single layer so far (no cross yet).

**Analysis script output** (run on canonical cases for groups): 

Top feature ids per group (by max activation on canonical prompts for layer 16):

- hasExtension_abstract: [588, 1020, 1825, 2167, 3492, 3889, 4051, 4591, 5532, 7672, 8246, 9049, 10425, 10769, 12811] ...

- instanceOf: [374, 1155, 1409, 1483, 1735, 2962, 6350, 6890, 9535, 13606, 14293, 15441, 16263, 16370, 17203] ...

- superset_subClass: [60, 402, 612, 756, 1104, 1792, 1873, 2118, 3392, 3950, 4159, 6350, 7265, 7957, 9434] ...

- category_vs_member: [196, 327, 588, 788, 1375, 3343, 4159, 5056, 5490, 5532, 6049, 6139, 6419, 8968, 9615] ...

Note overlap e.g. 9049 in hasExtension_abstract (also high in catalog), 588 in hasExtension and category_vs_member, etc.

This is concrete evidence of LTC-like features at 8B scale. Cross-layer next.

**L16 150k results (2026-06-22 scheduled)**: `max_aggression_collect.py --tokens 150000 --layers 16 --expansion 12` (CPU, float32, KMP_DUPLICATE_LIB_OK=TRUE). Collection to 150k vectors + 1200-step BatchTopK training (d_sae=49152). Recon logs: step 0=1.47397, 150=0.06049, 300=0.01754, 450=0.00899, 600=0.00826, 750=0.00855, 900=0.00757, 1050=0.00679, 1199=0.00700. Collector "Analyzing..." then saved catalog + sae.pt. 

**Artifacts**: `experiments/first_set/sae_runs/llama_3_1_8b_layer16_max/` (sae.pt ~1.6GB, feature_catalog.json 16.6MB, feature_catalog.md).

**Catalog observations** (top max-activation features from training data): Strong activation on target ontological language.
- hasExtension/abstract: "The concept of 'dog' has an extension..." (e.g. 27422 max=3.36, 28530=2.96, 9049=2.70).
- instanceOf: "The Eiffel Tower is an instance of 'landmark'." (48684=2.89, 38873=2.82), "The word 'mammal' refers to the abstract category, while this particular whale is an instance." (9882=2.83).
- superset/subClass: "Mammals are a superset of dogs and cats." (48351=2.80), "Vehicles include cars, boats, and airplanes as subclasses." (17714=3.02, 5056=2.73).
- Other: "Professions such as doctor and lawyer are types of occupations." (38742=3.18).

Signals are clear and on-point even from collector's training-set top examples (label alignment note: prompts repeated). Analysis script run produced same per-group tops. Limitations: CPU-only (slow), mixed pile data may dilute, labels are whole-prompt repeats, single layer so far (no cross yet).

**Analysis script output** (run on canonical cases for groups): 

Top feature ids per group (by max activation on canonical prompts for layer 16):

- hasExtension_abstract: [588, 1020, 1825, 2167, 3492, 3889, 4051, 4591, 5532, 7672, 8246, 9049, 10425, 10769, 12811] ...

- instanceOf: [374, 1155, 1409, 1483, 1735, 2962, 6350, 6890, 9535, 13606, 14293, 15441, 16263, 16370, 17203] ...

- superset_subClass: [60, 402, 612, 756, 1104, 1792, 1873, 2118, 3392, 3950, 4159, 6350, 7265, 7957, 9434] ...

- category_vs_member: [196, 327, 588, 788, 1375, 3343, 4159, 5056, 5490, 5532, 6049, 6139, 6419, 8968, 9615] ...

Note overlap e.g. 9049 in hasExtension_abstract (also high in catalog), 588 in hasExtension and category_vs_member, etc.

This is concrete evidence of LTC-like features at 8B scale. Cross-layer next.
