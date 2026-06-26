# Theory: Class Threads and the LLM-Tapestry Characteristic

## 1. The Core Problem: Ontological Impurity in LLMs

Large language models, particularly those trained primarily through next-token prediction on raw internet text, frequently fail to maintain clean distinctions between:

- **Intension** (the abstract idea or definition of a concept)
- **Extension** (the actual set of instances that belong to that concept)

This leads to various forms of **ontological impurity** — situations in which the model’s internal representations or generated outputs blur categories, confuse prototypes with full sets, or violate basic hierarchical and membership relationships. These impurities can manifest as incorrect inferences, inconsistent reasoning, and degraded performance on tasks that require precise categorization.

While post-training improves surface-level behavior, it does not reliably instill deep, consistent ontological structure.

## 2. The Class Thread Model

The **Class Thread** formalism was developed to provide a disciplined way of representing and maintaining ontological relationships. At its core, every concept is represented using two distinct nodes:

- An **abstract node**, representing the intension (the idea of the concept)
- An **extensional superset node**, representing the extension (the set of all instances)

These two nodes are connected by a dedicated relation called **`hasExtension`**.

A **Class Thread** is defined as a directed path with the following structure:

`()-[hasExtension]->()-[supersetOf]->()-[hasElement]->()`

with the number of sequential `-[supersetOf]->()` segments being any integer, including zero.

This structure enforces several important invariants:
- Clear separation between the abstract concept and its set of instances
- Consistent hierarchical relationships between categories
- Proper connection of individual instances to their categories

The three core relations in this framework are:
- `hasExtension` (abstract concept → its extensional set)
- `supersetOf` (and its inverse `subClassOf`)
- `hasElement` (and its inverse `instanceOf`)

## 3. The LLM-Tapestry Characteristic (LTC)

The **LLM-Tapestry Characteristic** is the hypothesis that certain fundamental ontological relations — particularly `hasExtension`, `subClassOf`, and `instanceOf` — should have relatively consistent internal representations within large language models.

In other words, we expect that the computational mechanisms the model uses to handle these relationships are not entirely ad-hoc or concept-specific, but instead exhibit some degree of regularity across different concepts and contexts. This consistency could appear as:
- Similar directions or features in activation space
- Reusable circuits or computational pathways
- Consistent patterns of interaction between features

If such consistent structure exists, it would be extremely valuable. It would suggest that models can be guided toward (or trained to maintain) a more disciplined internal ontology. This, in turn, could make it significantly easier to detect and correct ontological violations.

## 4. Research Questions

This line of work is organized around three central questions:

1. **Existence**: To what extent does something like the LTC already exist as an emergent property in current LLMs? Can we find evidence of consistent features or circuits corresponding to `hasExtension`, `subClassOf`, and `instanceOf`?

2. **Value of Enforcement**: Would deliberately strengthening or enforcing the LTC during training produce better models? In particular, would it improve performance on tasks requiring clean hierarchical and membership reasoning, and would it improve training efficiency?

3. **Trade-offs**: What are the potential downsides of enforcing stronger ontological structure? Could over-constraining the model reduce flexibility or harm other capabilities?

## 5. Connection to Mechanistic Interpretability

This project uses tools from **Mechanistic Interpretability** — particularly **Sparse Autoencoders** and **circuit analysis** — to investigate the internal representations and computations of language models.

Rather than only evaluating models through their external behavior, we aim to look inside the model to understand:
- Whether it develops relatively clean, disentangled features for concepts and instances
- Whether it implements consistent computational pathways for the core ontological relations
- Whether violations of Class Thread structure can be detected internally

The long-term goal is to move from *detecting* ontological problems to *preventing* them through better training methods and architectural choices.

## 6. Scope of This Repository

This repository focuses on the **mechanistic investigation** of Class Thread structure and the LLM-Tapestry Characteristic. It is distinct from earlier work on synthetic data generation and error repair. The emphasis here is on understanding what structure (if any) already exists inside models and how it might be strengthened.

Key themes:
- Feature discovery using Sparse Autoencoders
- Circuit analysis for relational structure
- Investigation of consistency across concepts (the LTC)
- Long-term goal of developing methods to enforce ontological hygiene during training
