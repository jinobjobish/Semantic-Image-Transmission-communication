import torch
from models.semantic_encoder import SemanticEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
enc = SemanticEncoder(bandwidth='low').to(device).eval()

# Load your trained SC1 weights if available
try:
    enc.load_state_dict(torch.load('checkpoints/encoder_sc1_final.pth', map_location=device, weights_only=True))
    print("Loaded SC1 weights")
except:
    print("Using random SC1 (untrained)")

black = torch.zeros(1, 3, 512, 512).to(device)
with torch.no_grad():
    out = enc(black)
print(f"Black encoding: min={out.min().item():.4f}, max={out.max().item():.4f}, mean={out.mean().item():.4f}")
if out.max() - out.min() < 0.1:
    print("✅ Black encoding is near‑constant – good for compression.")
else:
    print("❌ Black encoding varies a lot – train SC1 further.")