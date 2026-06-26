#!/usr/bin/env python3
"""
Simple watcher for max aggression Phase 1 artifacts.

Polls for new feature_catalog.json in sae_runs/*layer* dirs.
When found, runs analyze_consistency.py (if SAEs present) and prints summary.

Run in background while the big collection happens:
  python experiments/first_set/watch_and_analyze.py

It will fire the instant the first catalog from the 200k/300k runs appears.
"""

import time
from pathlib import Path
import subprocess
import json

BASE = Path("experiments/first_set/sae_runs")

seen = set()

print("Watching for new max-aggression catalogs (200k+ on L12/16/20 or 300k L16)...")

while True:
    for d in sorted(BASE.glob("*layer*")):
        cat = d / "feature_catalog.json"
        if cat.exists() and cat not in seen:
            seen.add(cat)
            print(f"\n=== NEW CATALOG: {cat} ===")
            meta = json.load(open(cat)).get("model_info", {})
            print("Meta:", meta)

            sae = d / "sae.pt"
            if sae.exists():
                print("SAE present — running full consistency analysis...")
                try:
                    subprocess.run(["python3", "experiments/first_set/analyze_consistency.py",
                                    "--run_base", str(d.parent / d.name.replace("layer", "layer").rsplit("_layer",1)[0]),
                                    "--layers", d.name.split("layer")[-1] if "layer" in d.name else "16"],
                                   check=False, timeout=300)
                except Exception as e:
                    print("Analysis run failed or partial:", e)
            else:
                print("No sae.pt yet — just catalog available.")

            print("=== Ready for next escalation (higher tokens, more layers, etc.) ===")

    time.sleep(30)  # poll every 30s
