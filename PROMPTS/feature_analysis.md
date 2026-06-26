# Feature Analysis Guidelines (Phase 1)

## What to look for
- Features that activate strongly and selectively on:
  - Category words: "mammal", "vehicle", "profession", "number", ...
  - Instance words: proper names, specific animals ("Fido"), specific models
- Features sensitive to relational phrases ("X is a type of Y", "all Xs have...", membership language)
- Features that do NOT fire on the surface form but on the *abstract* vs *extensional* distinction

## Process
1. After training, compute top-k activating dataset examples per feature (use analyze_features.py)
2. For the 200-500 highest "max activation" or "interesting" features, record:
   - 3-5 top activating texts (with context)
   - Apparent meaning (manual or LLM-assisted)
3. Tag features with preliminary labels:
   - "category-abstract"
   - "instance-specific"
   - "hierarchical-relation"
   - "other / polysemantic"
   - "noise / uninterpretable"

## Deliverable for Phase 1
A catalog (json + markdown) plus a short written assessment answering:
- Is there evidence of reasonably monosemantic features for abstract concepts vs. instances?
- Any early signs of consistency across different categories?

Keep all findings (positive and negative) in the notebook.
