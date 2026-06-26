"""
Visualization helpers for Phase 1 feature analysis.

Planned:
- Activation heatmaps
- Bar plots of feature sparsity / max act
- Simple token highlighting for max examples (future)
"""

from __future__ import annotations
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_feature_sparsity(summaries: List, top_n: int = 30, title: str = "Feature sparsity"):
    """Simple bar chart of sparsity for top-N features by max activation."""
    if not summaries:
        print("No summaries to plot.")
        return
    sorted_s = sorted(summaries, key=lambda s: s.max_activation, reverse=True)[:top_n]
    xs = [f"f{s.feature_idx}" for s in sorted_s]
    ys = [s.sparsity for s in sorted_s]

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(xs)), ys)
    plt.xticks(range(len(xs)), xs, rotation=90, fontsize=7)
    plt.ylabel("sparsity (frac > thresh)")
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def plot_max_activations(summaries: List, top_n: int = 30):
    sorted_s = sorted(summaries, key=lambda s: s.max_activation, reverse=True)[:top_n]
    vals = [s.max_activation for s in sorted_s]
    plt.figure(figsize=(10, 3.5))
    plt.bar(range(len(vals)), vals)
    plt.title("Max activation per feature (top N)")
    plt.tight_layout()
    return plt.gcf()
