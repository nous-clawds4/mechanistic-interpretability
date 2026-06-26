#!/usr/bin/env python3
"""
Environment Test for Mechanistic Interpretability Project (Phase 1)

This script verifies that the runtime environment is suitable for:
- Loading 7B-13B models via transformer_lens + Hugging Face
- Collecting residual stream activations
- Training/analyzing Sparse Autoencoders (sae_lens)

Run with:
    python3 test_environment.py

It is intentionally non-destructive and does not download large models unless --download-model is passed.
"""

import sys
import platform
import subprocess
from pathlib import Path

def check_command(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return None

def main():
    print("=" * 70)
    print("PHASE 1 ENVIRONMENT TEST")
    print("Project: Mechanistic Interpretability of Class Thread / LLM-Tapestry")
    print("=" * 70)

    # System info
    print("\n[1] System Information")
    print(f"  Python version: {sys.version}")
    print(f"  Python executable: {sys.executable}")
    print(f"  Platform: {platform.platform()}")
    print(f"  Machine: {platform.machine()}")
    cpu_brand = check_command("sysctl -n machdep.cpu.brand_string 2>/dev/null")
    if cpu_brand:
        print(f"  CPU: {cpu_brand}")
    print(f"  Working dir: {Path.cwd()}")

    # Disk
    disk = check_command("df -h . 2>/dev/null | tail -1")
    if disk:
        print(f"  Disk (cwd): {disk}")

    # Core Python libs
    print("\n[2] Core Python Libraries")
    core = {}
    for name in ["numpy", "matplotlib", "tqdm", "einops", "pandas", "scipy"]:
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "unknown")
            core[name] = ver
            print(f"  {name}: {ver}")
        except Exception as e:
            core[name] = None
            print(f"  {name}: MISSING ({type(e).__name__})")

    # PyTorch and device
    print("\n[3] PyTorch + Hardware Acceleration (critical for SAEs)")
    torch_info = {"installed": False, "version": None, "mps": False, "cuda": False, "device": "cpu"}
    try:
        import torch
        torch_info["installed"] = True
        torch_info["version"] = torch.__version__
        print(f"  torch: {torch.__version__}")

        # MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch_info["mps"] = True
            print("  MPS (Apple Silicon GPU): AVAILABLE")
            torch_info["device"] = "mps"
        else:
            print("  MPS (Apple Silicon GPU): NOT AVAILABLE (will fall back to CPU)")

        # CUDA (expect not on Mac)
        if torch.cuda.is_available():
            torch_info["cuda"] = True
            print(f"  CUDA: AVAILABLE (devices={torch.cuda.device_count()})")
            torch_info["device"] = "cuda"
        else:
            print("  CUDA: NOT AVAILABLE (expected on macOS)")

        print(f"  Recommended device for this run: {torch_info['device']}")
    except Exception as e:
        print(f"  torch: MISSING - {e}")

    # Mechanistic Interpretability stack
    print("\n[4] Mechanistic Interpretability Stack (Phase 1 requirements)")
    mi_libs = {
        "transformer_lens": "transformer_lens",
        "sae_lens": "sae_lens",
        "nnsight": "nnsight",
        "transformers": "transformers",
        "datasets": "datasets",
        "accelerate": "accelerate",
        "huggingface_hub": "huggingface_hub",
    }
    mi_status = {}
    for pretty, import_name in mi_libs.items():
        try:
            mod = __import__(import_name.replace("-", "_"))
            ver = getattr(mod, "__version__", "unknown")
            mi_status[pretty] = ver
            print(f"  {pretty}: {ver}")
        except Exception as e:
            mi_status[pretty] = None
            print(f"  {pretty}: NOT INSTALLED ({type(e).__name__})")

    # Other helpful
    print("\n[5] Optional / Nice-to-have")
    for name in ["jupyter", "notebook", "ipywidgets", "seaborn", "plotly"]:
        try:
            mod = __import__(name)
            print(f"  {name}: OK")
        except Exception:
            print(f"  {name}: MISSING (optional)")

    # Summary & recommendations
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    missing_critical = []
    if not torch_info["installed"]:
        missing_critical.append("torch")
    if mi_status.get("transformer_lens") is None:
        missing_critical.append("transformer_lens")
    if mi_status.get("sae_lens") is None:
        missing_critical.append("sae_lens")
    if mi_status.get("transformers") is None:
        missing_critical.append("transformers")

    if missing_critical:
        print("STATUS: INCOMPLETE - Critical packages missing for Phase 1.")
        print(f"Missing: {', '.join(missing_critical)}")
        print("\nTo set up (recommended order):")
        print("  pip3 install -r requirements.txt")
        print("\nOr install core stack manually:")
        print("  pip3 install torch torchvision torchaudio")
        print("  pip3 install transformers datasets accelerate")
        print("  pip3 install transformer-lens sae-lens")
        print("  pip3 install einops tqdm matplotlib pandas jupyter")
        print("\nNote on Apple Silicon (M-series):")
        print("  - Use device='mps' where supported.")
        print("  - Some transformer_lens / sae_lens ops may fall back to CPU or have limited support.")
        print("  - Consider smaller batch sizes and lower d_sae during initial experiments.")
        print("  - For very large activations, watch memory (M3 Ultra has lots of RAM).")
    else:
        print("STATUS: Core packages appear present.")
        print(f"  Primary device: {torch_info['device']}")

    if "3.1" in sys.version or "3.14" in sys.version:
        print("\nWARNING: Python 3.14+ is very new.")
        print("  Many scientific/ML wheels may not yet be available or may be unstable.")
        print("  If installs fail, consider using Python 3.11 or 3.12 via pyenv/conda.")

    print("\nNext steps after environment is ready:")
    print("  1. python3 -c 'import transformer_lens; print(transformer_lens.__version__)'")
    print("  2. Begin notebooks/phase1_feature_discovery.ipynb (or exploration.ipynb)")
    print("  3. Use Llama-3.1-8B-Instruct (or Mistral-7B) for initial runs.")
    print("=" * 70)

    # Return code for scripting
    if missing_critical:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
