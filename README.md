# Mechanistic Interpretability of Class Thread Structure

This repository explores whether large language models develop consistent internal representations of fundamental ontological relations, particularly those defined in the **Class Thread** model.

## Overview

Large language models often fail to cleanly distinguish between the *intension* (abstract idea) and *extension* (set of instances) of concepts. This can lead to subtle but important errors in hierarchical reasoning, membership judgments, and categorization.

The **Class Thread** formalism provides a structured way to represent these relationships using two nodes per concept (abstract + extensional) connected by specific directed relations:

- `hasExtension`
- `subClassOf` / `supersetOf`
- `instanceOf` / `hasElement`

This project investigates whether something like these relations — which we refer to as the **LLM-Tapestry Characteristic (LTC)** — emerges naturally inside current models, and whether it can be strengthened or enforced.

## Goals

- Use tools from **Mechanistic Interpretability** (Sparse Autoencoders, circuit analysis, causal interventions) to search for internal representations of Class Thread structure.
- Understand whether these representations are consistent across different concepts and layers.
- Explore whether enforcing more consistent ontological structure during training improves model behavior and efficiency.

## Current Status

This is an early-stage research project. Work is currently focused on **exploratory analysis** using 7B–13B class models. The initial phase centers on feature discovery using Sparse Autoencoders.

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the current experimental plan.

## Repository Structure

```
mechanistic-interpretability/
├── README.md
├── CLAUDE.md                 # Instructions for working with Claude Code
├── THEORY.md                 # Conceptual background
├── EXPERIMENTS.md            # Detailed experimental plan
├── master-prompt.md          # Full instructions for Claude Code
├── src/                      # Source code for experiments
├── notebooks/                # Exploration and analysis notebooks
├── PROMPTS/                  # Reusable prompts
└── requirements.txt
```

## Getting Started

This project is designed to be worked on collaboratively with **Claude Code**.

1. Read [`CLAUDE.md`](CLAUDE.md) for working guidelines.
2. Read [`THEORY.md`](THEORY.md) for conceptual background.
3. Read [`EXPERIMENTS.md`](EXPERIMENTS.md) for the current experimental direction.
4. Start working according to the instructions in `CLAUDE.md`.

## Key Concepts

- **Class Thread**: A directed path of the form `hasExtension → (subClassOf)* → instanceOf`
- **LLM-Tapestry Characteristic (LTC)**: The hypothesis that core ontological relations have relatively consistent internal representations in language models.
- **Intension vs Extension**: The distinction between the abstract definition of a concept and the set of things that actually belong to it.

## Contributing

This is primarily a research exploration. When working with Claude Code:

- Follow the workflow described in `CLAUDE.md`
- Work iteratively and confirm major steps before proceeding
- Prioritize clarity and reproducibility

## License

This repository is currently unlicensed as it is in an early research stage.

## Acknowledgments

This work builds on ideas from Mechanistic Interpretability (particularly Sparse Autoencoders and circuit analysis) and the Class Thread model for ontological hygiene in knowledge representation.
