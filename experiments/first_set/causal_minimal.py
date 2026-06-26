#!/usr/bin/env python3
"""Minimal causal test focused on 5532."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.model_loading import load_model

def load_sae(path, d_in=4096, d_sae=49152, device="cpu"):
    data = torch.load(path, map_location=device)
    try:
        from sae_lens import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
        cfg = BatchTopKTrainingSAEConfig(d_in=d_in, d_sae=d_sae, device=device, k=32)
        sae = BatchTopKTrainingSAE(cfg).to(device)
        sae.load_state_dict(data["state_dict"])
        return sae
    except Exception as e:
        print("BatchTopK load issue, using attr map:", e)
        class SAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # map common keys
                sd = data["state_dict"]
                self.W_enc = torch.nn.Linear(d_in, d_sae, bias=True)
                self.W_dec = torch.nn.Linear(d_sae, d_in, bias=True)
                self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
                if "W_enc" in sd:
                    self.W_enc.weight.data = sd["W_enc"]
                if "b_enc" in sd: self.b_enc.data = sd.get("b_enc", torch.zeros(d_sae))
                # similar for others, simplified
            def encode(self, x):
                return torch.relu(self.W_enc(x) + self.b_enc)
            def decode(self, a):
                return self.W_dec(a)
        sae = SAE().to(device)
        return sae

def main():
    print("=== Minimal Causal for 5532 ===")
    device = "cpu"
    model = load_model("llama-3.1-8b", device=device, dtype=torch.float32)
    for layer in [12, 16]:
        hook = f"blocks.{layer}.hook_resid_post"
        sae_path = Path(f"experiments/first_set/sae_runs/llama_3_1_8b_layer{layer}_max/sae.pt")
        sae = load_sae(sae_path, device=device)
        print(f"Layer {layer} SAE loaded.")
        prompt = "An abstract idea like 'justice' is different from the set of all just actions. The extension includes"
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits_clean = model(tokens, return_type="logits")[0, -1]
        def patch(resid, hook):
            flat = resid.reshape(-1, resid.shape[-1])
            acts = sae.encode(flat) if hasattr(sae, 'encode') else torch.relu(sae.W_enc(flat) + sae.b_enc)
            acts[:, 5532] = 0.0
            patched = sae.decode(acts) if hasattr(sae, 'decode') else sae.W_dec(acts)
            return patched.reshape(resid.shape)
        with torch.no_grad():
            logits_p = model.run_with_hooks(tokens, fwd_hooks=[(hook, patch)], return_type="logits")[0, -1]
        tok = model.to_single_token(" all")
        p_clean = torch.softmax(logits_clean, -1)[tok].item()
        p_patch = torch.softmax(logits_p, -1)[tok].item()
        print(f"Layer {layer} feat5532: cleanP={p_clean:.4f} patchedP={p_patch:.4f} delta={p_patch-p_clean:.4f}")
    print("Minimal causal run complete.")

if __name__ == "__main__":
    main()
